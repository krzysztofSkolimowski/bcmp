import tkinter as tk
from bcmp import *


def is_matrix(val):
    return hasattr(val[0], '__iter__')


class SimpleTableInput(tk.Frame):
    def __init__(self, parent, rows, columns, values=None):
        tk.Frame.__init__(self, parent)

        self._entry = {}
        self.rows = rows
        self.columns = columns

        # register a command to use for validation
        vcmd = (self.register(self._validate), "%P")

        # create the table of widgets
        for row in range(self.rows):
            for column in range(self.columns):
                index = (row, column)
                e = tk.Entry(self, validate="key", validatecommand=vcmd)
                e.grid(row=row, column=column)
                self._entry[index] = e

        # adjust column weights so they all expand equally
        for column in range(self.columns):
            self.grid_columnconfigure(column, weight=0, pad=0)

        # designate a final, empty row to fill up any extra space
        self.grid_rowconfigure(rows, weight=0, pad=0)

        if values is not None:
            if not is_matrix(values):
                for c in range(columns):
                    index = (0, c)
                    self._entry[index].insert(0, values[c])
            else:
                row_index = 0
                for v in values:
                    for c in range(columns):
                        index = (row_index, c)

                        self._entry[index].insert(0, v[c])
                    row_index += 1


    def get(self):
        '''Return a list of lists, containing the data in the table'''
        result = []
        for row in range(self.rows):
            current_row = []
            for column in range(self.columns):
                index = (row, column)
                current_row.append(self._entry[index].get())
            result.append(current_row)
        return result

    def _validate(self, P):
        '''Perform input validation.

        Allow only an empty value, or a value that can be converted to a float
        '''
        if P.strip() == "":
            return True

        try:
            f = float(P)
        except ValueError:
            self.bell()
            return False
        return True


class Grid(tk.Frame):
    def __init__(self, parent, model):
        # todo - comment

        tk.Frame.__init__(self, parent)

        self.n = SimpleTableInput(self, 1, 8, model["n"])
        self.u = SimpleTableInput(self, 8, 3, model["u"])
        self.m = SimpleTableInput(self, 1, 8, model["m"])
        self.p_1 = SimpleTableInput(self, 8, 8, model["p"][0])
        self.p_2 = SimpleTableInput(self, 8, 8, model["p"][1])
        self.p_3 = SimpleTableInput(self, 8, 8, model["p"][2])
        self.k = SimpleTableInput(self, 1, 3, model["k"])
        self.epsilon = SimpleTableInput(self, 1, 1)

        # self.load_button = tk.Button(self, text="Load json", command=self.on_submit)
        self.submit_button = tk.Button(self, text="Submit", command=self.on_submit)

        # self.n.pack(side="top", fill="none", expand=True)
        # self.load_button.pack(side="top")
        tk.Label(self, text="n").pack(ipadx="2px", ipady="2px")
        self.n.pack(ipadx="2px", ipady="2px")
        tk.Label(self, text="u").pack(ipadx="2px", ipady="2px")
        self.u.pack(ipadx="2px", ipady="2px")
        tk.Label(self, text="m").pack(ipadx="2px", ipady="2px")
        self.m.pack(ipadx="2px", ipady="2px")
        tk.Label(self, text="p").pack(ipadx="2px", ipady="2px")
        self.p_1.pack(ipadx="2px", ipady="2px")
        self.p_2.pack(ipadx="2px", ipady="2px")
        self.p_3.pack(ipadx="2px", ipady="2px")
        tk.Label(self, text="k").pack(ipadx="2px", ipady="2px")
        self.k.pack(ipadx="2px", ipady="2px")
        tk.Label(self, text="epsilon").pack(ipadx="2px", ipady="2px")
        self.epsilon.pack(ipadx="2px", ipady="2px")

        self.submit_button.pack(side="bottom")

    def on_submit(self):
        print(self.n.get())
        model = load_data()
        result = "RESULT"
        # tk.Label(self, text=result).pack(anchor="W")




def main():
    root = tk.Tk()
    model = load_data()
    Grid(root, model).pack(side="top", fill="both", expand=True)
    root.mainloop()


if __name__ == '__main__':
    main()
