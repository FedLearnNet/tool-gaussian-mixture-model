from pyfedappwrap.engine.runtime import FedDBEngine

from app import GmmAPP

engine = FedDBEngine()
engine.register(GmmAPP())

if __name__ == '__main__':
    engine.start()
    engine.wait_until_stop()
