# Helpers
def display_title(title, h=1):
    match h:
        case 1:
            cprint(title, "blue", "on_light_grey")
        case 2:
            cprint(title, "black", "on_cyan")
        case 3:
            cprint(title, "black", "on_light_yellow")
        case _:
            cprint(title, "white", "on_light_magenta")

def display_title2(title, h=1):
    match h:
        case 1:
            print(colored(title, "blue"))
        case 2:
            print(colored(title, "red"))
        case 3:
            print(colored(title, "green"))
        case _:
            print(colored(title, "yellow"))

display_title("Date")
display(datetime.now(pytz.timezone("Europe/Brussels")).strftime("%Y-%m-%d %H:%M:%S"))
