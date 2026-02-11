import pyfiglet
from termcolor import colored

def gen_banner():    
    banner = pyfiglet.figlet_format("STONKS")
    colored_banner = colored(banner, color="blue", attrs=["bold"])
    
    print(colored_banner)