from time import time

class Block:
      def __init__(self,index,previous_hahs,transactions,proof,time=time()):
          self.index=index
          self.previous_hash=previous_hahs
          self.transactions=transactions
          self.proof=proof
          self.time=time
