from oracle.monk.monk_model import MonkModel

monk = MonkModel(num_dataset=1)

monk.train(epochs=300)
monk.print_results(plot=True)