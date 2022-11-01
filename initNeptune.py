from mimetypes import init
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from neptune.new.types import File
from neptuneCredentials import Cred
class Neptune:
    def __init__(self, projectName="", api_token=""):
        self.parameters={}


    def startRun(self,parameters={}):
        self.run = neptune.init(project=Cred.project, api_token=Cred.api_token)
        self.run["model/parameters"] = parameters
        return self.getcallback()
    
    def stopRun(self):
        self.run.stop()

    def getcallback(self):                
        neptune_cbk = NeptuneCallback(run=self.run, base_namespace='metrics')
        return neptune_cbk

    def resetTags(self):
        self.run["sys/tags"].clear()

    def addTags(self,tags=[]):
        self.run["sys/tags"].add(tags)