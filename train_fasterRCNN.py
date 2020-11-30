import torch
import os
import utils


def main():
    try:
        current_file_dir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
    except NameError:
        current_file_dir = os.getcwd().replace('\\', '/')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    frame_dir = current_file_dir + '/data/training_images/'
    label_dir = current_file_dir + '/data/'
    package_dataset = utils.PackageDataset(frame_dir=frame_dir, label_dir=label_dir,
                                           transform=utils.get_transform(train=True))

    # define training data loader
    data_loader_train = torch.utils.data.DataLoader(package_dataset, batch_size=4, shuffle=True,
                                                    collate_fn=utils.collate_fn)

    # get model instance
    model = utils.get_model_instance(num_classes=2)
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by 2/3 every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.3)

    # let's train it for xx epochs
    num_epochs = 1
    for epoch in range(num_epochs):
        print('Epoch {}'.format(epoch))
        # train for one epoch, printing every 10 iterations
        utils.train(model, optimizer, data_loader_train)
        # update the learning rate
        lr_scheduler.step()

    torch.save(model.state_dict(), current_file_dir + '/model.pt')


if __name__ == '__main__':
    main()
