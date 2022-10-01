import multiprocessing as mp
import signal

# just contains 1 function, which is currently not used.
# it was used in the VGG implementation, but not necessary for ours.


def setup_multiprocessing(num_processes=8):
    """ 
    Sets up multiprocessing for preparing the training data.
    This was implemented by the authors of the VGG model.
    """
    # signal.signal() allows custom handlers to be executed when a signal is received
    # signal.SIGINT: interrupt from keyboard, default behavior is to terminate the process
    # with signal.SIG_IGN, I believe the default behavior is changed to ignoring the keyboard interrupts
    def init_worker():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    # This code is causing an error, not sure how to fix it.
    # try:
    #     global pool
    #     pool = None
    #     pool.terminate()
    # except:
    #     pass

    if num_processes > 0:
        pool = mp.Pool(processes=num_processes, initializer=init_worker)
    else:
        pool = None
    return pool
