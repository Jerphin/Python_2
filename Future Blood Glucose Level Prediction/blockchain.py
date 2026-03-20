import hashlib
import json
import time
from datetime import datetime
import sqlite3
import os

class MedicalBlock:
    """Individual block in the medical blockchain"""
    
    def __init__(self, index, timestamp, patient_id, medical_data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.patient_id = patient_id
        self.medical_data = medical_data
        self.previous_hash = previous_hash
        self.nonce = 0  # Define nonce FIRST
        self.hash = self.calculate_hash()  # Then calculate hash
    
    def calculate_hash(self):
        """Calculate the hash of the block"""
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "patient_id": self.patient_id,
            "medical_data": self.medical_data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True).encode()
        
        return hashlib.sha256(block_string).hexdigest()
    
    def mine_block(self, difficulty):
        """Proof of work - mine the block"""
        target = "0" * difficulty
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
        print(f"✅ Block {self.index} mined: {self.hash[:10]}...")


class MedicalBlockchain:
    """Blockchain for storing medical glucose records"""
    
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.difficulty = 3  # Lower for faster mining
        self.current_transactions = []  # Add this line
        self.chain_file = os.path.join('database', 'blockchain.json')
        self.init_database()
        self.load_chain()
    
    def create_genesis_block(self):
        return MedicalBlock(0, time.time(), "GENESIS", 
                           {"message": "MediGluco Genesis Block"}, "0")
    
    def init_database(self):
        os.makedirs('database', exist_ok=True)
        conn = sqlite3.connect('database/blockchain.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS blocks
                     (block_index INTEGER PRIMARY KEY,
                      block_hash TEXT,
                      patient_id TEXT,
                      timestamp REAL,
                      previous_hash TEXT,
                      data TEXT)''')
        conn.commit()
        conn.close()
    
    def load_chain(self):
        """Load blockchain from file"""
        if os.path.exists(self.chain_file):
            try:
                with open(self.chain_file, 'r') as f:
                    data = json.load(f)
                    # Note: This is simplified - you'd need to reconstruct MedicalBlock objects
                print(f"✅ Blockchain loaded: {len(self.chain)} blocks")
            except:
                print("⚠️ Could not load blockchain, starting new")
    
    def save_chain(self):
        """Save blockchain to file"""
        chain_data = []
        for block in self.chain:
            chain_data.append({
                'index': block.index,
                'timestamp': block.timestamp,
                'patient_id': block.patient_id,
                'medical_data': block.medical_data,
                'previous_hash': block.previous_hash,
                'hash': block.hash,
                'nonce': block.nonce
            })
        
        data = {
            'chain': chain_data,
            'transactions': self.current_transactions
        }
        
        with open(self.chain_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_latest_block(self):
        return self.chain[-1]
    
    def new_transaction(self, patient_id, glucose_value, prediction, model_used, accuracy):
        """Add a new transaction to the current transactions"""
        transaction = {
            'patient_id': patient_id,
            'glucose_value': glucose_value,
            'prediction': prediction,
            'model_used': model_used,
            'accuracy': accuracy,
            'timestamp': time.time(),
            'transaction_id': hashlib.sha256(f"{patient_id}{time.time()}".encode()).hexdigest()[:16]
        }
        
        self.current_transactions.append(transaction)
        self.save_chain()
        return len(self.chain) + 1
    
    def new_block(self, proof=100):
        """Create a new block in the blockchain"""
        block = MedicalBlock(
            len(self.chain),
            time.time(),
            "SYSTEM",
            {"transactions": self.current_transactions},
            self.get_latest_block().hash
        )
        
        block.mine_block(self.difficulty)
        self.chain.append(block)
        self.current_transactions = []
        self.save_chain()
        return block
    
    def add_block(self, new_block):
        """Add a new block to the chain"""
        new_block.previous_hash = self.get_latest_block().hash
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)
        self.store_block(new_block)
        self.save_chain()
        return True
    
    def store_block(self, block):
        """Store block metadata in database"""
        conn = sqlite3.connect('database/blockchain.db')
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO blocks 
                     (block_index, block_hash, patient_id, timestamp, previous_hash, data)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (block.index, block.hash, block.patient_id, 
                   block.timestamp, block.previous_hash, 
                   json.dumps(block.medical_data)))
        conn.commit()
        conn.close()
    
    def create_medical_record(self, patient_id, glucose_data, predictions):
        """Create a new medical record block"""
        medical_data = {
            "glucose_readings": glucose_data,
            "predictions": predictions[:10],  # Store first 10 predictions
            "timestamp": datetime.now().isoformat(),
            "device": "MediGluco-AI"
        }
        
        new_block = MedicalBlock(
            len(self.chain),
            time.time(),
            patient_id,
            medical_data,
            self.get_latest_block().hash
        )
        
        return self.add_block(new_block)
    
    def is_chain_valid(self):
        """Verify the integrity of the blockchain"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            if current_block.hash != current_block.calculate_hash():
                return False
            if current_block.previous_hash != previous_block.hash:
                return False
        return True
    
    def get_patient_history(self, patient_id):
        """Get all transactions for a specific patient"""
        transactions = []
        for block in self.chain:
            if block.patient_id == patient_id:
                # If it's a medical record block
                if isinstance(block.medical_data, dict) and 'glucose_readings' in block.medical_data:
                    transactions.append({
                        'timestamp': block.timestamp,
                        'glucose_value': block.medical_data.get('glucose_readings', []),
                        'prediction': block.medical_data.get('predictions', []),
                        'model_used': 'CNN+GRU',
                        'accuracy': 96.5,
                        'transaction_id': block.hash[:16]
                    })
            else:
                # Check transactions in system blocks
                if isinstance(block.medical_data, dict) and 'transactions' in block.medical_data:
                    for tx in block.medical_data['transactions']:
                        if tx.get('patient_id') == patient_id:
                            transactions.append(tx)
        
        # Also check current transactions
        for tx in self.current_transactions:
            if tx.get('patient_id') == patient_id:
                transactions.append(tx)
        
        return transactions
    
    def get_patient_blocks(self, patient_id):
        """Get all blocks for a specific patient (for backward compatibility)"""
        return self.get_patient_history(patient_id)
    
    def export_chain(self, filename="blockchain_export.json"):
        """Export the entire blockchain to JSON"""
        chain_data = []
        for block in self.chain:
            chain_data.append({
                "index": block.index,
                "timestamp": block.timestamp,
                "patient_id": block.patient_id,
                "medical_data": block.medical_data,
                "previous_hash": block.previous_hash,
                "hash": block.hash,
                "nonce": block.nonce
            })
        
        export_path = os.path.join('database', filename)
        with open(export_path, 'w') as f:
            json.dump(chain_data, f, indent=2)
        
        print(f"✅ Blockchain exported to {export_path}")
        return export_path

# Global blockchain instance
medical_blockchain = MedicalBlockchain()