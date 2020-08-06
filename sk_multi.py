from skmultiflow.data import SEAGenerator
from skmultiflow.trees import HoeffdingTree
from skmultiflow.evaluation import EvaluatePrequential

# create a stream
stream = SEAGenerator()
tree = HoeffdingTree()

# Training a Hoeffding Tree for Classification
correctness_dist = []

nb_iters = 10000

# ========== Method2 to picture the accuracy ===========
evaluator = EvaluatePrequential(show_plot = True, max_samples = nb_iters)
evaluator.evaluate(stream=stream, model=tree)

# prepare the stream for use
stream.prepare_for_use()

# get a data sample
# X, Y = stream.next_sample()

# print(X, Y)

# ========== Method1 to picture the accuracy ==========

# for i in range(nb_iters):
#     # get the next sample
#     X, Y = stream.next_sample()
#     # predict Y using the tree
#     prediction = tree.predict(X)
    
#     # check the prediciton
#     if Y == prediction:
#         correctness_dist.append(1)
#     else:
#         correctness_dist.append(0)
    
#     # update the tree
#     tree.partial_fit(X, Y)

# import matplotlib.pyplot as plt

# time = [i for i in range(1, nb_iters)]

# accuracy = [sum(correctness_dist[:i])/len(correctness_dist[:i]) for i in range(1, nb_iters)]

# plt.plot(time, accuracy)
# plt.show()

