import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback


class Neptune:
    def getcallback(name = 'Untilted', tags = []):
        run = neptune.init(project='cailen01/branchingDNN', name = name, api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2NDJjNzA1Yi1iMTA5LTRjYzgtYTAyNS1lMDE1NTFkZjQ2NDEifQ==', tags=tags)
        neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')
        return neptune_cbk