import yaml


class Config:
    def __init__(self):
        with open('conf.yaml') as f:
            self.doc = yaml.safe_load(f)
        #print('Initialized config with doc = ', self.doc)

    def get(self, key:str):
        return self.doc.get(key)

    @property
    def backend_host(self):
        return self.doc.get('backend_host')

    @property
    def rest_api_port(self):
        return self.doc.get('rest_api_port')

    @property
    def grpc_api_port(self):
        return self.doc.get('grpc_api_port')

    @property
    def model_name(self):
        return self.doc.get('model_name')

