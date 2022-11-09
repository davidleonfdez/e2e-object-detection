import yaml


class Config:
    def __init__(self):
        with open('conf.yaml') as f:
            self.doc = yaml.safe_load(f)

    def get(self, key:str):
        return self.doc.get(key)

    @property
    def backend_url(self):
        return self.doc.get('backend_url')
