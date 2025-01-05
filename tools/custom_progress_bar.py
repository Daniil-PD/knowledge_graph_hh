
class Progress_bar():
    def __init__(self, total:int=100, bar_size:int=80, on_line:bool = True, label:str=None, final_label=None):
        self.total = total
        self.bar_size = bar_size
        self.on_line = on_line
        self.label = label
        if not label is None and final_label is None:
            self.final_label = "Complete!"
        

    def __call__(self, i):
        percent = i / self.total
        left_char_count = int(percent*self.bar_size)
        
        string_out = f"""{self.label + " " if self.label else ""}[{"#"*left_char_count}{" "*(self.bar_size-left_char_count)}] {int(percent*100):3}%"""
              
        
        
        if i == self.total:
            if not self.final_label is None:
                string_out += " " + self.final_label
            string_out += "\n"
        elif self.on_line:
            string_out += "\r"
        else:
            string_out += "\n"

        print(string_out, end="")
        


# if __name__ == "__main__":
#     import time
#     pb = Progress_bar(total=100, label="Test")
#     for i in range(101):
#         pb(i)
#         time.sleep(0.01)
