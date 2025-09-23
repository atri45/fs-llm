import logging
import random
import math
import pickle
import os

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric import padding as rsa_padding 
from cryptography.hazmat.primitives import padding as symmetric_padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from federatedscope.core.message import Message

logger = logging.getLogger(__name__)

class AnonymousRouter:
    def __init__(self, client_id, cfg):
        """
        初始化匿名路由器。

        Args:
            client_id (int): 当前客户端的 ID。
            cfg: 全局配置对象。
        """
        self.client_id = client_id
        self.cfg = cfg
        
        # peers 和 all_client_ids 将在稍后由 Client 注入
        self.peers = []
        self.all_client_ids = []
        
        # --- 密钥管理 ---
        # 1. 生成自己的密钥对
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=1024
        )
        self.public_key = self.private_key.public_key()
        
        # 2. 准备一个字典来存储所有人的公钥，初始时只有自己
        self.public_keys = {self.client_id: self.public_key}

        # --- 路径构建超参数 ---
        self.path_len_a = 2.0
        self.path_len_b = 1.0
        self.steepness_s = 3.0
        
        # 用于注入 Engine，以便在最终接收时调用
        self.engine = None

    def _get_path_decision(self, current_path_len):
        """
        根据公式(1)（对数版本），决定下一跳是目标还是中继。
        """
        n = len(self.all_client_ids)
        i = current_path_len
        if n < 2: return True
        
        # 对数期望路径长度
        i_0 = self.path_len_a * math.log(n - 1) + self.path_len_b
        # 陡峭度
        k = self.steepness_s / i_0
        
        p_target = 1 / (1 + math.exp(-k * (i - i_0)))
        
        return random.random() < p_target

    def build_anonymous_route(self, original_content, target_client_id, original_state):
        """
        构建匿名路径，对路由信封使用混合加密。
        """
        try:
            # --- 1. 构建路径 ---
            path = []
            path_nodes = {self.client_id}
            
            # 防御性检查
            if target_client_id not in self.all_client_ids:
                logger.error(f"Build route failed: Target #{target_client_id} is not in known clients {self.all_client_ids}.")
                return None, None

            max_path_length = len(self.all_client_ids)
            while len(path) < max_path_length:
                candidate_pool = [pid for pid in self.all_client_ids if pid not in path_nodes]
                if not candidate_pool: break
                next_hop_id = random.choice(candidate_pool)
                path.append(next_hop_id)
                path_nodes.add(next_hop_id)
                if next_hop_id == target_client_id: break
            
            if not path or path[-1] != target_client_id:
                path = [target_client_id] # Fallback to direct path

            # logger.info(f"Client #{self.client_id}: Built anonymous path to {target_client_id}: {path}")

            # --- 2. 准备明文数据载荷 ---
            plaintext_payload = {
                "content": original_content,
                "state": original_state
            }
            
            # --- 3. 对路由信息进行洋葱式的【混合加密】 ---
            # a. 准备最内层的路由信息 (给最终接收者)
            inner_most_route_info = {"next_hop": -1}
            current_payload_to_encrypt = pickle.dumps(inner_most_route_info)
            
            # b. 从后向前，为路径上的每一个节点创建加密层
            for i in range(len(path) - 1, -1, -1):
                encrypt_for_node_id = path[i]
                
                encrypting_key = self.public_keys.get(encrypt_for_node_id)
                if not encrypting_key:
                    logger.error(f"Public key for hop Client #{encrypt_for_node_id} not found!")
                    return None, None

                # --- 执行混合加密 ---
                aes_key = os.urandom(32) # AES-256
                iv = os.urandom(16)      # AES block size is 128 bits

                # 1. 用 AES 加密【当前信封】
                cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
                encryptor = cipher.encryptor()
                padder = symmetric_padding.PKCS7(algorithms.AES.block_size).padder()
                padded_data = padder.update(current_payload_to_encrypt) + padder.finalize()
                aes_encrypted_payload = encryptor.update(padded_data) + encryptor.finalize()

                # 2. 用 RSA 公钥加密 AES 密钥
                rsa_encrypted_aes_key = encrypting_key.encrypt(
                    aes_key,
                    rsa_padding.OAEP(
                        mgf=rsa_padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )

                # 3. 构造新的、外层的信封结构
                outer_envelope = {
                    "rsa_encrypted_aes_key": rsa_encrypted_aes_key,
                    "iv": iv,
                    "encrypted_inner_payload": aes_encrypted_payload
                }

                # 4. 序列化，作为下一次循环的输入
                current_payload_to_encrypt = pickle.dumps(outer_envelope)
            encrypted_route_envelope = current_payload_to_encrypt

            # --- 4. 准备最终要发送的消息内容 ---
            final_message_content = {
                "encrypted_route_envelope": list(encrypted_route_envelope),
                "payload": plaintext_payload
            }
                
            first_hop_id = path[0]
            return first_hop_id, final_message_content

        except Exception as e:
            logger.exception(f"Unhandled exception in build_anonymous_route for target {target_client_id}: {e}")
            return None, None

    def handle_anonymous_message(self, message: Message):
        """
        处理分离了路由和数据的消息，对路由信封进行【混合解密】。
        """
        try:
            message_content = message.content
            # 1. 分离出加密的路由信封和明文载荷
            encrypted_route_envelope_list = message_content["encrypted_route_envelope"]
            current_envelope_bytes = bytes(encrypted_route_envelope_list)
            payload = message_content["payload"]

            # 2. Unpickle 最外层信封结构
            outer_envelope = pickle.loads(current_envelope_bytes)
            # 3. 执行混合解密
            # a. 用自己的私钥解密 AES 密钥
            aes_key = self.private_key.decrypt(
                outer_envelope["rsa_encrypted_aes_key"],
                rsa_padding.OAEP(
                    mgf=rsa_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            # b. 用解密出的 AES 密钥解密内层信封
            cipher = Cipher(algorithms.AES(aes_key), modes.CBC(outer_envelope["iv"]))
            decryptor = cipher.decryptor()
            padded_inner_envelope = decryptor.update(outer_envelope["encrypted_inner_payload"]) + decryptor.finalize()
            unpadder = symmetric_padding.PKCS7(algorithms.AES.block_size).unpadder()
            # inner_envelope_bytes 是给下一跳的、完整的、pickle过的内层信封
            inner_envelope_bytes = unpadder.update(padded_inner_envelope) + unpadder.finalize()
            # 4. Unpickle 内层信封，以获取下一跳信息
            inner_info = pickle.loads(inner_envelope_bytes)
            next_hop_id = inner_info.get("next_hop")

        except Exception as e:
            logger.exception(f"Client #{self.client_id}: Decryption/parsing of route envelope failed for message from sender {message.sender}: {e}")
            return None

        # 5. 根据路由信息采取行动
        if next_hop_id == -1:
            # 我是最终接收者
            return ('aggregate', {
                'content': payload['content'],
                'state': payload['state']
            })
        else:
            # 我是中继节点
            # 准备要转发给下一跳的新消息体
            forward_content = {
                # 内层信封现在是外层信封
                "encrypted_route_envelope": list(inner_envelope_bytes),
                # 载荷原封不动地继续传递
                "payload": payload
            }
            return ('forward', {
                'receiver': [next_hop_id],
                'content': forward_content,
                'state': message.state # state 可以沿用上一跳的
            })