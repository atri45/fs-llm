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
            key_size=2048
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
        使用“随机前向选择”算法来构建匿名路径。
        """
        # --- 1. 路径选择 ---
        path = []
        # `path_nodes` 集合用于快速查找路径中已有的节点，避免重复
        path_nodes = {self.client_id}
        current_node_id = self.client_id
        
        while True:
            # a. 确定下一跳的“候选池”
            #    候选者是所有客户端，除了【已经】在路径上的
            candidate_pool = [
                pid for pid in self.all_client_ids 
                if pid not in path_nodes
            ]
            
            # b. 如果没有候选者了（例如，网络中只有2个客户端），则直接结束
            if not candidate_pool:
                break
                
            # c. 在候选池中，进行等概率随机选择
            next_hop_id = random.choice(candidate_pool)
            
            # d. 将选中的节点加入路径
            path.append(next_hop_id)
            path_nodes.add(next_hop_id)
            
            # e. 检查是否选中了最终目标
            if next_hop_id == target_client_id:
                # 选中了目标，路径构建结束
                break
            
            # f. 如果选中的是中继，则继续循环
            current_node_id = next_hop_id
            
        # 如果因为某种原因（例如，候选池为空）导致路径为空，
        # 那么直接发送给目标
        if not path:
            path.append(target_client_id)

        logger.info(f"Client #{self.client_id}: Built basic anonymous path to {target_client_id}: {path}")

        # --- 2. 洋葱式混合加密 (这部分逻辑与我们之前的最终版本完全相同) ---
        # a. 准备最内层 payload
        inner_most_payload_bytes = pickle.dumps({
            "content": original_content, "state": original_state
        })
        
        current_encrypted_payload = inner_most_payload_bytes
        
        # b. 从后向前，为路径上的每一个节点创建加密层
        for i in range(len(path) - 1, -1, -1):
            encrypt_for_node_id = path[i]
            next_hop_id = path[i+1] if i + 1 < len(path) else -1
            
            encrypting_key = self.public_keys.get(encrypt_for_node_id)
            if not encrypting_key:
                logger.error(f"Public key for hop Client #{encrypt_for_node_id} not found!")
                return None, None

            # --- 混合加密【当前 payload】 ---
            aes_key = os.urandom(32)
            iv = os.urandom(16)
            
            # 1. 用 AES 加密
            cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
            encryptor = cipher.encryptor()
            padder = symmetric_padding.PKCS7(algorithms.AES.block_size).padder()
            padded_data = padder.update(current_encrypted_payload) + padder.finalize()
            aes_encrypted_payload = encryptor.update(padded_data) + encryptor.finalize()
            
            # 2. 用 RSA 公钥加密 AES 密钥
            rsa_encrypted_aes_key = encrypting_key.encrypt(aes_key, rsa_padding.OAEP(
                    mgf=rsa_padding.MGF1(algorithm=hashes.SHA256()), 
                    algorithm=hashes.SHA256(), 
                    label=None
                ))

            # 3. 构造新的、外层的 payload，结构统一
            outer_payload = {
                "next_hop": next_hop_id,
                "rsa_encrypted_aes_key": rsa_encrypted_aes_key,
                "iv": iv,
                "encrypted_inner_payload": aes_encrypted_payload
            }

            # 4. 将这个字典序列化，作为下一次循环的【待加密】数据
            current_encrypted_payload = pickle.dumps(outer_payload)
            
        first_hop_id = path[0]
        return first_hop_id, list(current_encrypted_payload)

    def build_anonymous_route1(self, original_content, target_client_id, original_step):
        """
        一个统一的接口，负责构建、加密并将匿名消息发送给第一跳。
        """
        # 1. 路径选择
        path = [] # [hop_1_id, hop_2_id, ..., target_id]
        current_len = 1
        
        # 排除自己和最终目标，得到可用的中继节点列表
        possible_relays = [pid for pid in self.peers if pid != target_client_id]
        
        while not self._get_path_decision(current_len):
            if not possible_relays: # 如果没有可用的中继了，只能直接发给目标
                break
            # 随机选择一个中继
            relay_id = random.choice(possible_relays)
            path.append(relay_id)
            possible_relays.remove(relay_id) # 确保中继不重复
            current_len += 1

        path.append(target_client_id)
        logger.info(f"Client #{self.client_id}: Built anonymous path to {target_client_id}: {path}")

        # 2. 洋葱式加密
        # a. 【最内层 payload】: 这是给【最终接收者 target_client_id】看的
        #    它不包含路由信息，只包含真正的数据。
        inner_most_payload_bytes = pickle.dumps({
            "content": original_content,
            "state": original_step
        })

        # `current_encrypted_payload` 始终是【内层的、已被加密】的数据
        current_encrypted_payload = inner_most_payload_bytes
        
        # b. 从后向前，为【路径上的每一个节点】创建加密层
        #    倒序遍历路径: target -> hop_2 -> hop_1
        for i in range(len(path) - 1, -1, -1):
            encrypt_for_node_id = path[i]
            next_hop_id = path[i+1] if i + 1 < len(path) else -1
            
            # 获取要为其加密的那个节点的公钥
            encrypting_key = self.public_keys.get(encrypt_for_node_id)
            if not encrypting_key:
                 logger.error(f"Public key for hop Client #{encrypt_for_node_id} not found!")
                 return None, None

            # --- 混合加密【当前 payload】 ---
            aes_key = os.urandom(32)
            iv = os.urandom(16)
            
            # 1. 用 AES 加密
            cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
            encryptor = cipher.encryptor()
            padder = symmetric_padding.PKCS7(algorithms.AES.block_size).padder()
            padded_data = padder.update(current_encrypted_payload) + padder.finalize()
            aes_encrypted_payload = encryptor.update(padded_data) + encryptor.finalize()
            
            # 2. 用 RSA 公钥加密 AES 密钥
            rsa_encrypted_aes_key = encrypting_key.encrypt(aes_key, rsa_padding.OAEP(
                    mgf=rsa_padding.MGF1(algorithm=hashes.SHA256()), 
                    algorithm=hashes.SHA256(), 
                    label=None
                ))

            # 3. 构造新的、外层的 payload，结构统一
            outer_payload = {
                "next_hop": next_hop_id,
                "rsa_encrypted_aes_key": rsa_encrypted_aes_key,
                "iv": iv,
                "encrypted_inner_payload": aes_encrypted_payload
            }
            
            # 4. 将这个字典序列化，作为下一次循环的【待加密】数据
            current_encrypted_payload = pickle.dumps(outer_payload)

        # 最终发送给第一跳的，就是最外层的 pickle 字节串
        first_hop_id = path[0]
        return first_hop_id, list(current_encrypted_payload)
        
    def handle_anonymous_message(self, message: Message):
        """
        【分离式加密版】只解密路由信封。
        """
        payload_as_list_of_ints = message.content
        
        try:
            # 1. 恢复最外层的 pickle 字节串
            outer_payload_bytes = bytes(payload_as_list_of_ints)
            
            # 2. Unpickle，得到外层数据结构
            outer_data = pickle.loads(outer_payload_bytes)
            
            rsa_encrypted_aes_key = outer_data["rsa_encrypted_aes_key"]
            iv = outer_data["iv"]
            encrypted_inner_payload = outer_data["encrypted_inner_payload"]
            next_hop_id = outer_data["next_hop"]

            # 3. 混合解密
            aes_key = self.private_key.decrypt(rsa_encrypted_aes_key, rsa_padding.OAEP(
                    mgf=rsa_padding.MGF1(algorithm=hashes.SHA256()), 
                    algorithm=hashes.SHA256(), 
                    label=None
                ))
            cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
            decryptor = cipher.decryptor()
            padded_decrypted_payload = decryptor.update(encrypted_inner_payload) + decryptor.finalize()
            unpadder = symmetric_padding.PKCS7(algorithms.AES.block_size).unpadder()
            inner_payload_bytes = unpadder.update(padded_decrypted_payload) + unpadder.finalize()
            
        except Exception as e:
            logger.error(f"Client #{self.client_id}: Decryption/parsing failed: {e}", exc_info=True)
            return None

        if next_hop_id == -1:
            # --- 我是最终接收者 ---
            # 此时 `inner_payload_bytes` 是最内层的、包含原始数据的 pickle
            final_data = pickle.loads(inner_payload_bytes)
            return ('aggregate', {
                'content': final_data['content'],
                'state': final_data['state']
            })
        else:
            # --- 我是中继节点 ---
            # 此时 `inner_payload_bytes` 是需要转发给下一跳的、
            # 仍然是 pickle 过的、加密的 payload
            return ('forward', {
                'receiver': [next_hop_id],
                'content': list(inner_payload_bytes),
                'state': message.state
            })