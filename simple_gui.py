from bcmp import *
import tkinter as tk


def main():
    root = tk.Tk()

    class GUI(tk.Frame):
        def __init__(self, parent):
            super(GUI, self).__init__()

            tk.Label(parent, text="model").pack(side=tk.LEFT, ipadx="2px", ipady="2px")
            self.json_input = tk.Text(parent, height=50, width=100)
            self.json_input.pack(side=tk.LEFT)

            tk.Label(parent, text="result").pack(side=tk.RIGHT, ipadx="2px", ipady="2px")
            self.result_input = tk.Text(parent, height=50, width=100)
            self.result_input.pack(side=tk.RIGHT)

            submit_button = tk.Button(parent, text="SUBMIT\n>>>", height=50, command=self.on_submit)
            submit_button.pack(side=tk.TOP)

            with open('model.json') as f:
                self.json_input.insert(tk.END, f.read())
            parent.mainloop()

        def on_submit(self):
            model_str = self.json_input.get(1.0, tk.END)
            model = json.loads(model_str)
            q_n = GNetwork(model)

            K = q_n.compute_average_k()
            T = q_n.compute_average_t(K)
            W = q_n.compute_average_w(T)
            Q = q_n.compute_average_q(W)

            result = ""
            result += "Row: Queueing system number" + "\n"
            result += 'Column: Request class' + "\n"
            result += 'K: ' + "\n"
            result += str(K) + "\n"
            result += 'T: ' + "\n"
            result += str(T) + "\n"
            result += 'W: ' + "\n"
            result += str(W) + "\n"
            result += 'Q: ' + "\n"
            result += str(Q) + "\n"

            self.result_input.delete(1.0, tk.END, )
            self.result_input.insert(tk.END, result)

    GUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
