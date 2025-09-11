import threading

from federatedscope.core.message import Message
from federatedscope.core.proto import gRPC_comm_manager_pb2, \
    gRPC_comm_manager_pb2_grpc


class gRPCComServeFunc(gRPC_comm_manager_pb2_grpc.gRPCComServeFuncServicer):
    def __init__(self):
        # 使用一个字典来存储每个 sender 的【最新】消息
        self.latest_msg_from_sender = {}
        # 需要一个锁来保护这个共享字典
        self.lock = threading.Lock()
        # 使用一个 Condition 变量来高效地实现阻塞 receive
        self.new_msg_condition = threading.Condition(self.lock)

    def sendMessage(self, request, context):
        temp_msg = Message()
        temp_msg.parse(request.msg)
        sender_id = temp_msg.sender
        
        with self.lock:
            # 直接用新消息【覆盖】这个 sender 的旧消息
            self.latest_msg_from_sender[sender_id] = request
            self.new_msg_condition.notify()

        return gRPC_comm_manager_pb2.MessageResponse(msg='ACK')

    def receive(self):
        """
        一个【阻塞】的接收方法。
        """
        with self.lock:
            # 如果字典里没有任何消息，就等待
            while not self.latest_msg_from_sender:
                # new_msg_condition.wait() 会原子性地释放锁并休眠
                # 当被 notify() 唤醒时，它会重新获取锁
                self.new_msg_condition.wait()
            sender_id, received_request = self.latest_msg_from_sender.popitem()
            
        return received_request

    def receive_nowait(self):
        """
        一个【非阻塞】的接收方法。
        """
        with self.lock:
            if not self.latest_msg_from_sender:
                return None
            
            # 将当前的所有最新消息打包成一个列表返回
            received_requests = list(self.latest_msg_from_sender.values())
            self.latest_msg_from_sender.clear()
        
        return received_requests