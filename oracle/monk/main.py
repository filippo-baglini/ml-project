from oracle.monk.monk_model import MonkModel

monk = MonkModel(num_dataset=3)

monk.train()
monk.print_results()