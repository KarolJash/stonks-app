import sys
import time
import itertools
import threading

def create_loading_animation(stop_event):
    for c in itertools.cycle(["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]):
        if stop_event.is_set():
            break
        sys.stdout.write(f'\rLoading {c} ')
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\rDone!     \n\n')

def start_animation(text):
    stop_loading = threading.Event()

    print(f"{text} ")
    t = threading.Thread(target=create_loading_animation, args=(stop_loading,))
    t.start()

    return [stop_loading, t]

def end_animation(stop_loading, t):
    stop_loading.set()
    t.join()