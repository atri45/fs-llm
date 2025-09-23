import threading
import grpc
from concurrent import futures
import logging
import torch.distributed as dist

from collections import deque

from federatedscope.core.proto import gRPC_comm_manager_pb2, \
    gRPC_comm_manager_pb2_grpc
from federatedscope.core.gRPC_server import gRPCComServeFunc, anonymous_gRPCComServeFunc
from federatedscope.core.message import Message

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StandaloneCommManager(object):
    """
    The communicator used for standalone mode
    """
    def __init__(self, comm_queue, monitor=None):
        self.comm_queue = comm_queue
        self.neighbors = dict()
        self.monitor = monitor  # used to track the communication related
        # metrics

    def receive(self):
        # we don't need receive() in standalone
        pass

    def add_neighbors(self, neighbor_id, address=None):
        self.neighbors[neighbor_id] = address

    def get_neighbors(self, neighbor_id=None):
        address = dict()
        if neighbor_id:
            if isinstance(neighbor_id, list):
                for each_neighbor in neighbor_id:
                    address[each_neighbor] = self.get_neighbors(each_neighbor)
                return address
            else:
                return self.neighbors[neighbor_id]
        else:
            # Get all neighbors
            return self.neighbors

    def send(self, message):
        # All the workers share one comm_queue
        self.comm_queue.append(message)


class StandaloneDDPCommManager(StandaloneCommManager):
    """
    The communicator used for standalone mode with multigpu
    """
    def __init__(self, comm_queue, monitor=None, id2comm=None):
        super().__init__(comm_queue, monitor)
        self.id2comm = id2comm
        self.device = "cuda:{}".format(dist.get_rank())

    def _send_model_para(self, model_para, dst_rank):
        for v in model_para.values():
            t = v.to(self.device)
            dist.send(tensor=t, dst=dst_rank)

    def send(self, message):
        is_model_para = message.msg_type == 'model_para'
        is_evaluate = message.msg_type == 'evaluate'
        if self.id2comm is None:
            # client to server
            if is_model_para:
                model_para = message.content[1]
                message.content = (message.content[0], {})
                self.comm_queue.append(message) if isinstance(
                    self.comm_queue, deque) else self.comm_queue.put(message)
                self._send_model_para(model_para, 0)
            else:
                self.comm_queue.append(message) if isinstance(
                    self.comm_queue, deque) else self.comm_queue.put(message)
        else:
            receiver = message.receiver
            if not isinstance(receiver, list):
                receiver = [receiver]
            if is_model_para or is_evaluate:
                model_para = message.content
                message.content = {}
            for idx, each_comm in enumerate(self.comm_queue):
                for each_receiver in receiver:
                    if each_receiver in self.neighbors and \
                            self.id2comm[each_receiver] == idx:
                        each_comm.put(message)
                        break
                if is_model_para or is_evaluate:
                    for each_receiver in receiver:
                        if each_receiver in self.neighbors and \
                                self.id2comm[each_receiver] == idx:
                            self._send_model_para(model_para, idx + 1)
                            break
        download_bytes, upload_bytes = message.count_bytes()
        self.monitor.track_upload_bytes(upload_bytes)


class gRPCCommManager(object):
    """
        The implementation of gRPCCommManager is referred to the tutorial on
        https://grpc.io/docs/languages/python/
    """
    def __init__(self, host='0.0.0.0', port='50050', client_num=2, cfg=None):
        self.host = host
        self.port = port
        self.channel_options = [
            ("grpc.max_send_message_length", cfg.distribute.grpc_max_send_message_length),
            ("grpc.max_receive_message_length", cfg.distribute.grpc_max_receive_message_length),
            ("grpc.enable_http_proxy", cfg.distribute.grpc_enable_http_proxy),
        ]

        if cfg.distribute.grpc_compression.lower() == 'deflate':
            self.comp_method = grpc.Compression.Deflate
        elif cfg.distribute.grpc_compression.lower() == 'gzip':
            self.comp_method = grpc.Compression.Gzip
        else:
            self.comp_method = grpc.Compression.NoCompression

        if cfg.federate.anonymous_routing:
            self.server_funcs = anonymous_gRPCComServeFunc()
        else:
            self.server_funcs = gRPCComServeFunc()
        self.grpc_server = self.serve(max_workers=client_num,
                                      host=host,
                                      port=port,
                                      options=self.channel_options)
        self.neighbors = {} # 只存储地址
        self.stubs = {}     # 缓存已建立的连接 (stub)
        self.lock = threading.Lock() # 用于保护 stubs 字典的线程安全
        self.monitor = None  # used to track the communication related metrics

    def serve(self, max_workers, host, port, options):
        """
        This function is referred to
        https://grpc.io/docs/languages/python/basics/#starting-the-server
        """
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            compression=self.comp_method,
            options=options)
        gRPC_comm_manager_pb2_grpc.add_gRPCComServeFuncServicer_to_server(
            self.server_funcs, server)
        server.add_insecure_port("{}:{}".format(host, port))
        server.start()

        return server

    def add_neighbors(self, neighbor_id, address):
        if isinstance(address, dict):
            self.neighbors[neighbor_id] = '{}:{}'.format(address['host'], address['port'])
        elif isinstance(address, str):
            self.neighbors[neighbor_id] = address
        else:
            raise TypeError(...)
        logger.info(f"Neighbor #{neighbor_id} at {self.neighbors[neighbor_id]} has been registered.")

    def get_neighbors(self, neighbor_id=None):
        address = dict()
        if neighbor_id:
            if isinstance(neighbor_id, list):
                for each_neighbor in neighbor_id:
                    address[each_neighbor] = self.get_neighbors(each_neighbor)
                return address
            else:
                return self.neighbors[neighbor_id]
        else:
            # Get all neighbors
            return self.neighbors

    def get_stub(self, receiver_id):
        """
        懒加载获取到某个接收者的 Stub。
        如果连接已缓存，直接返回；否则，创建、缓存并返回。
        """
        # 先在无锁情况下快速检查，提高性能
        if receiver_id in self.stubs:
            return self.stubs[receiver_id]

        # 如果未找到，进入线程安全的创建流程
        with self.lock:
            # 再次检查，防止在等待锁的过程中其他线程已经创建了它
            if receiver_id in self.stubs:
                return self.stubs[receiver_id]

            receiver_address = self.neighbors.get(receiver_id)
            if not receiver_address:
                logger.warning(f"Address for neighbor #{receiver_id} not found.")
                return None

            # 创建长连接 Channel
            channel = grpc.insecure_channel(
                receiver_address,
                compression=self.comp_method,
                options=self.channel_options
            )
            
            # 创建 Stub 并缓存
            stub = gRPC_comm_manager_pb2_grpc.gRPCComServeFuncStub(channel)
            self.stubs[receiver_id] = stub
            logger.info(f"Lazily established a persistent gRPC connection to neighbor #{receiver_id} at {receiver_address}")
            return stub

    def _send(self, receiver_id, message, blocking=False):
        """
        使用 get_stub 获取连接，然后发送。
        """
        stub = self.get_stub(receiver_id)
        if stub is None:
            return

        request = message.transform(to_list=True)

        try:
            future = stub.sendMessage.future(request, timeout=15.0)
        
            if blocking:
                # 如果是阻塞模式，则等待 RPC 完成
                future.result() # 这会阻塞直到收到服务器的 ACK
                
        except grpc.RpcError as error:
            logger.error(f"gRPC call to #{receiver_id} failed with unrecoverable error: {error}")

    def send(self, message):
        # --- 检查消息类型，决定是否阻塞 ---
        blocking_msg_types = ['join_in', 'fedspeed_setup', 'fedspeed_init_package', 'fedspeed_ready', 'early_stop_vote' ,'sync_model_para', 'aggregated_model_para']
        is_blocking = message.msg_type in blocking_msg_types

        receiver = message.receiver
        if receiver is not None:
            if not isinstance(receiver, list):
                receiver = [receiver]
            for each_receiver_id in receiver:
                if each_receiver_id in self.neighbors:
                    self._send(each_receiver_id, message, blocking=is_blocking)
        else:
            # 广播给所有邻居
            for each_receiver_id in self.neighbors:
                self._send(each_receiver_id, message, blocking=is_blocking)

    def receive(self):
        received_msg = self.server_funcs.receive()
        message = Message()
        message.parse(received_msg.msg)
        return message

    def receive_nowait(self):
        received_msg = self.server_funcs.receive_nowait()
        if received_msg is None:
            return None
        message = Message()
        message.parse(received_msg.msg)
        return message
    
    def stop(self):
        """
        在程序结束时，优雅地关闭服务器。
        """
        logger.info("Stopping gRPC server...")
        self.grpc_server.stop(grace=1.0)