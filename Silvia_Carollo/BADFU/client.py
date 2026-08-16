from server import Server
from config import TARGET_LABEL, TRIGGER_SIZE, TRIGGER_VAL

class Client:
    def __init__(self, id, model_state_dict, data):
        self.id = id
        self.model_state_dict = model_state_dict
        self.data = data 

    def train_model():
        pass 
    
    def request_unlearning(self, server: Server):
        # Request unlearning from the server
        server.unlearn(self.id)


class BadClient(Client):
    def __init__(self, id, model_state_dict, data):
        super().__init__(id, model_state_dict, data)
        

        self.poisoned_data = self.poison_data(copy.deepcopy(data)) 


    def poison_data(self, data):
        pass 

    def add_backdoor_trigger(self, data):
        # 

    def prepare_backdoor_data(self, data): #it's adviced to pass a deep copy of the data 
        data.data[ :, -TRIGGER_SIZE:, -TRIGGER_SIZE:] = TRIGGER_VAL
        data.targets[:] = TARGET_LABEL
        return data

    def get_training_data(self):
        int n_good_samples = len(self.data)
        int n_backdoor_samples = len(self.backdoor_data)

        while n_good_samples > 0 or n_backdoor_samples > 0: 
            if n_good_samples > 0:

                n_good_samples -= 1
            if n_backdoor_samples > 0:
                yield self.backdoor_data[n_backdoor_samples - 1]
                n_backdoor_samples -= 1
        return 
