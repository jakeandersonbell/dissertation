"""This file holds the additional functions used in modelling"""

import torch
import os


def shuffle_data(X, y, perm=False):
    # Shuffle data, can give a permutation for an existing shuffle
    if str(perm) == 'False':
        perm = torch.randperm(len(X[0]))
    train_X1 = X[0][perm][:-val_size * 2]
    train_X2 = X[1][perm][:-val_size * 2]
    train_X3 = X[2][perm][:-val_size * 2]
    train_y = y[perm][:-val_size * 2]

    val_X1 = X[0][perm][-val_size * 2:-val_size]
    val_X2 = X[1][perm][-val_size * 2:-val_size]
    val_X3 = X[2][perm][-val_size * 2:-val_size]
    val_y = y[perm][-val_size * 2:-val_size]

    return train_X1, train_X2, train_X3, train_y, val_X1, val_X2, val_X3, val_y, perm


def shuffle_train(train_X1, train_X2, train_X3, train_y):
    perm = torch.randperm(len(train_X1))
    train_X1 = train_X1[perm]
    train_X2 = train_X2[perm]
    train_X3 = train_X3[perm]
    train_y = train_y[perm]
    return train_X1, train_X2, train_X3, train_y


def shuffle_val(val_X1, val_X2, val_X3, val_y):
    perm = torch.randperm(len(val_X1))
    val_X1 = val_X1[perm]
    val_X2 = val_X2[perm]
    val_X3 = val_X3[perm]
    val_y = val_y[perm]
    return val_X1, val_X2, val_X3, val_y


# Forward pass in a function so it can be used to also calculate validation acc/loss
def fwd_pass(X1, X2, X3, y, train=False):
    if train:
        net.zero_grad()
    outputs = net(X1.to(device), X2.to(device), X3.to(device))
    matches = [torch.round(i) == j for i, j in zip(outputs, y)]

    acc = matches.count(True) / len(matches)
    y = y.unsqueeze(1)
    loss = loss_function(outputs, y)
    loss_class_weighted = loss * weight_
    loss = loss_class_weighted.mean()

    if train:
        loss.backward()
        optimizer.step()
    return acc, loss


def test(test_X1, test_X2, test_X3, test_y, size=128):
    # Get a slice starting at a random point
    test_X1, test_X2, test_X3, test_y = shuffle_val(test_X1, test_X2, test_X3, test_y)
    X1, X2, X3 = test_X1[:size].view(-1, 1, 128, 128), test_X2[:size].view(-1, 1, 64, 64), test_X3[:size]
    y = test_y[:size]
    with torch.no_grad():
        val_acc, val_loss = fwd_pass(X1.to(device), X2.to(device), X3.to(device), y.to(device))
    return val_acc, val_loss


def train(net, EPOCHS, BATCH_SIZE, LR, LYR, START_NODES, cnn_node, cnn_layer, xy, target=0):
    MODEL_NAME = f"model-{int(time.time())}"
    print(MODEL_NAME)
    train_X1, train_X2, train_X3, train_y, val_X1, val_X2, val_X3, val_y = xy
    with open("model.log", 'a') as f:
        n_epochs_stop = 20
        epochs_no_improve = 0
        total_epochs = 0

        min_val_loss = 100

        for epoch in range(EPOCHS):
            train_X1, train_X2, train_X3, train_y = shuffle_train(train_X1, train_X2, train_X3, train_y)
            total_epochs += 1

            for i in range(0, len(train_X1),
                           BATCH_SIZE):  # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
                #                 print(f"{i}:{i+BATCH_SIZE}")
                batch_X = [train_X1[i:i + BATCH_SIZE].view(-1, 1, 128, 128),
                           train_X2[i:i + BATCH_SIZE].view(-1, 1, 64, 64),
                           train_X3[i:i + BATCH_SIZE]]
                batch_y = train_y[i:i + BATCH_SIZE]

                net.zero_grad()

                acc, loss = fwd_pass(batch_X[0].to(device), batch_X[1].to(device), batch_X[2].to(device),
                                     batch_y.to(device), train=True)

                if i % 32 == 0:
                    val_acc, val_loss = test(val_X1, val_X2, val_X3, val_y)

                    f.write(
                        f"{MODEL_NAME},{round(time.time(), 3)},{epoch},{round(float(acc), 2)},{round(float(loss), 2)},"
                        f"{round(float(val_acc), 2)},{round(float(val_loss), 2)}\n")

            print(f"Epoch: {epoch}. Loss: {loss}. Accuracy: {acc}. Val loss: {val_loss}")
            print(optimizer.param_groups[0]['lr'])
            scheduler.step(val_loss)
            if val_loss < min_val_loss:
                # Remove existing model before saving
                existing_model = [i for i in os.listdir('models/new') if MODEL_NAME in i]

                if len(existing_model):
                    os.remove(os.path.join('models/new', existing_model[0]))

                torch.save(net.state_dict(), os.path.join('models', 'new', str(MODEL_NAME) + '_' + str(
                    round(float(val_loss), 4)) + ".pickle"))
                min_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if val_loss < target or epochs_no_improve == n_epochs_stop:
                break
    #     print(MODEL_NAME, EPOCHS, BATCH_SIZE, LR, LYR, START_NODES)
    return MODEL_NAME, EPOCHS, total_epochs, LR, LYR, START_NODES, cnn_node, cnn_layer, loss, val_loss


def get_select_features(not_arr, xy, test=False):
    new_xy = xy.copy()
    # Function to remove features from training data
    indices = torch.tensor([i for i in range(0, 43) if i not in not_arr])
    new_xy[2] = torch.index_select(xy[2], 1, indices)
    if not test:
        new_xy[6] = torch.index_select(xy[6], 1, indices)
    return new_xy