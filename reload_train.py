import torch
from dataset import SEGAN_Dataset, emphasis
from hparams import hparams
from model import Generator, Discriminator
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
# from adabelief_pytorch import AdaBelief
# from ranger21 import ranger21
from radam import RAdam

""" 使用可使用的gpu """


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


if __name__ == "__main__":

    device = try_gpu()
    print(device)

    para = hparams()

    os.makedirs(para.save_path, exist_ok=True)

    generator_file = "save/model/proportional_distribution/G_8_0.3678.pkl"
    discriminator_file = "save/model/proportional_distribution/G_8_0.3678.pkl"
    load_generator = torch.load(generator_file, map_location='cpu')
    load_discriminator = torch.load(discriminator_file, map_location='cpu')

    #  创建生成器
    generator = Generator()
    generator.load_state_dict(load_generator['model'])
    generator = generator.to(device)

    #  创建鉴别器
    discriminator = Discriminator()
    discriminator.load_state_dict(load_discriminator['model'])
    discriminator = discriminator.to(device)



    #  定义数据集，随机分布
    m_dataset = SEGAN_Dataset(para)

    #  定义数据集，按比例分布或者按同种分布
    # para.train_mode = 1
    # m_dataset_eng = SEGAN_Dataset(para)
    # para1 = hparams()
    # para1.train_mode = 2
    # m_dataset_chi = SEGAN_Dataset(para1)

    #  创建优化器RMSprop
    # g_optimizer = torch.optim.RMSprop(generator.parameters(), lr=0.0001)
    # d_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=0.0001)

    #  创建优化器AdaBelief
    """g_optimizer = AdaBelief(generator.parameters(), lr=2e-4, eps=1e-16, weight_decay=0, betas=(0.5, 0.999),
                            weight_decouple=True,
                            rectify=True)
    # d_optimizer = AdaBelief(discriminator.parameters(), lr=2e-4, eps=1e-16, weight_decay=0, betas=(0.5, 0.999),
                            weight_decouple=True,
                            rectify=True)
    """
    #  创建优化器Ranger21
    # g_optimizer = ranger21.Ranger21(generator.parameters(), lr=0.0001, num_epochs=para.n_epoch,
    #                                 num_warmup_iterations=100,
    #                                 warmdown_active=False,
    #                                 betas=(0.5, 0.999),
    #                                 weight_decay=0,
    #                                 eps = 1e-16,
    #                                 num_batches_per_epoch=int(len(m_dataset.clean_files)/para.batch_size))
    # d_optimizer = ranger21.Ranger21(discriminator.parameters(), lr=0.0001, num_epochs=para.n_epoch,
    #                                 num_warmup_iterations=100,
    #                                 warmdown_active=False,
    #                                 betas=(0.5, 0.999),
    #                                 weight_decay=0,
    #                                 eps = 1e-16,
    #                                 num_batches_per_epoch=int(len(m_dataset.clean_files)/para.batch_size))

    #  创建优化器RAdam
    g_optimizer = RAdam(generator.parameters(), lr=0.0001, eps=1e-16, betas=(0.5, 0.999))
    d_optimizer = RAdam(discriminator.parameters(), lr=0.0001, eps=1e-16, betas=(0.5, 0.999))

    #  获取ref_batch，随机分布
    ref_batch = m_dataset.ref_batch(para.ref_batch_size)
    ref_batch = Variable(ref_batch)
    ref_batch = ref_batch.to(device)

    #  获取ref_batch，按比例分布
    # ref_batch1 = m_dataset_eng.ref_batch(para.ref_batch_size/2)
    # ref_batch2 = m_dataset_chi.ref_batch(para.ref_batch_size/2)
    # ref_batch = np.concatenate((ref_batch1, ref_batch2), axis=0)
    # ref_batch = Variable(ref_batch)
    # ref_batch = ref_batch.to(device)

    #  获取ref_batch，同种分布
    # ref_batch1 = m_dataset_eng.ref_batch(para.ref_batch_size/2)
    # ref_batch1 = Variable(ref_batch1)
    # ref_batch1 = ref_batch1.to(device)
    # ref_batch2 = m_dataset_chi.ref_batch(para.ref_batch_size/2)
    # ref_batch2 = Variable(ref_batch2)
    # ref_batch2 = ref_batch2.to(device)

    #  定义dataloader，随机分布
    m_dataloader = DataLoader(m_dataset, batch_size=para.batch_size, shuffle=True, num_workers=0)

    #  定义dataloader，按比例分布
    # m_dataloader1 = DataLoader(m_dataset_eng, batch_size=para.batch_size / 2, shuffle=True, num_workers=0)
    # m_dataloader2 = DataLoader(m_dataset_chi, batch_size=para.batch_size / 2, shuffle=True, num_workers=0)

    #  定义dataloader，同种分布
    # m_dataloader1 = DataLoader(m_dataset_eng, batch_size=para.batch_size, shuffle=True, num_workers=0)
    # m_dataloader2 = DataLoader(m_dataset_chi, batch_size=para.batch_size, shuffle=True, num_workers=0)

    loss_d_all = 0
    loss_g_all = 0
    n_step = 0
    dataset_mode = 0  # 控制数据集的分布，为1表示同种分布，中文与英文数据集交替训练，为0表示随机分布、按比例分布

    for epoch in range(para.n_epoch):

        g_cond_loss = 0
        noisy_loss = 0

        if dataset_mode == 0:

            # 随机分布
            for i_batch, sample_batch in enumerate(m_dataloader):
                batch_clean = sample_batch[0]
                batch_noisy = sample_batch[1]

                # 等比例分布
                # for i_batch, data in enumerate(zip(m_dataloader1, m_dataloader2)):
                #     batch_clean = np.concatenate((data[0][0], data[1][0]), axis=0)
                #     batch_noisy = np.concatenate((data[0][1], data[1][1]), axis=0)

                #  声明为变量
                batch_clean = Variable(batch_clean)
                batch_noisy = Variable(batch_noisy)

                batch_clean = batch_clean.to(device)
                batch_noisy = batch_noisy.to(device)

                batch_z = nn.init.normal_(torch.Tensor(batch_clean.size(0), para.size_z[0], para.size_z[1]))
                batch_z = Variable(batch_z)
                batch_z = batch_z.to(device)

                #  Discriminator的训练
                discriminator.zero_grad()
                train_batch = Variable(torch.cat([batch_clean, batch_noisy], axis=1))
                outputs = discriminator(train_batch, ref_batch)
                clean_loss = 0.5 * torch.mean(outputs - 1.0) ** 2

                generated_outputs = generator(batch_noisy, batch_z)
                outputs = discriminator(torch.cat((generated_outputs, batch_noisy), dim=1), ref_batch)
                noisy_loss = 0.5 * torch.mean(outputs ** 2)

                d_loss = clean_loss + noisy_loss
                d_loss.backward()
                d_optimizer.step()

                #  Generator的训练
                generator.zero_grad()
                generated_outputs = generator(batch_noisy, batch_z)
                gen_noisy_pair = torch.cat((generated_outputs, batch_noisy), dim=1)
                outputs = discriminator(gen_noisy_pair, ref_batch)

                g_loss = 0.5 * torch.mean((outputs - 1.0) ** 2)
                l1_dist = torch.abs(torch.add(generated_outputs, torch.neg(batch_clean)))
                g_cond_loss = 100 * torch.mean(l1_dist)
                g_loss = g_loss + g_cond_loss

                g_loss.backward()
                g_optimizer.step()

                print("Epoch %d:%d d clean loss %.4f, d noisy loss %.4f, g loss %.4f, g conditional loss %.4f" % (
                    epoch + 1,
                    i_batch,
                    clean_loss,
                    noisy_loss,
                    g_loss,
                    g_cond_loss))
        # elif dataset_mode == 1:

        # 同种分布
        # for i_batch, data in enumerate(zip(m_dataloader1, m_dataloader2)):
        #     batch_clean = data[0][0]
        #     batch_noisy = data[0][1]
        #
        #     #  声明为变量
        #     batch_clean = Variable(batch_clean)
        #     batch_noisy = Variable(batch_noisy)
        #
        #     batch_clean = batch_clean.to(device)
        #     batch_noisy = batch_noisy.to(device)
        #
        #     batch_z = nn.init.normal_(torch.Tensor(batch_clean.size(0), para.size_z[0], para.size_z[1]))
        #     batch_z = Variable(batch_z)
        #     batch_z = batch_z.to(device)
        #
        #     #  Discriminator的训练
        #     discriminator.zero_grad()
        #     train_batch = Variable(torch.cat([batch_clean, batch_noisy], axis=1))
        #     outputs = discriminator(train_batch, ref_batch1)
        #     clean_loss = 0.5 * torch.mean(outputs - 1.0) ** 2
        #
        #     generated_outputs = generator(batch_noisy, batch_z)
        #     outputs = discriminator(torch.cat((generated_outputs, batch_noisy), dim=1), ref_batch1)
        #     noisy_loss = 0.5 * torch.mean(outputs ** 2)
        #
        #     d_loss = clean_loss + noisy_loss
        #     d_loss.backward()
        #     d_optimizer.step()
        #
        #     #  Generator的训练
        #     generator.zero_grad()
        #     generated_outputs = generator(batch_noisy, batch_z)
        #     gen_noisy_pair = torch.cat((generated_outputs, batch_noisy), dim=1)
        #     outputs = discriminator(gen_noisy_pair, ref_batch1)
        #
        #     g_loss = 0.5 * torch.mean((outputs - 1.0) ** 2)
        #     l1_dist = torch.abs(torch.add(generated_outputs, torch.neg(batch_clean)))
        #     g_cond_loss = 100 * torch.mean(l1_dist)
        #     g_loss = g_loss + g_cond_loss
        #
        #     g_loss.backward()
        #     g_optimizer.step()
        #
        #     print("Epoch %d:%d d clean loss %.4f, d noisy loss %.4f, g loss %.4f, g conditional loss %.4f" % (
        #         epoch + 1,
        #         i_batch,
        #         clean_loss,
        #         noisy_loss,
        #         g_loss,
        #         g_cond_loss))
        #
        #     batch_clean = data[1][0]
        #     batch_noisy = data[1][1]
        #     #  声明为变量
        #     batch_clean = Variable(batch_clean)
        #     batch_noisy = Variable(batch_noisy)
        #
        #     batch_clean = batch_clean.to(device)
        #     batch_noisy = batch_noisy.to(device)
        #
        #     batch_z = nn.init.normal_(torch.Tensor(batch_clean.size(0), para.size_z[0], para.size_z[1]))
        #     batch_z = Variable(batch_z)
        #     batch_z = batch_z.to(device)
        #
        #     #  Discriminator的训练
        #     discriminator.zero_grad()
        #     train_batch = Variable(torch.cat([batch_clean, batch_noisy], axis=1))
        #     outputs = discriminator(train_batch, ref_batch2)
        #     clean_loss = 0.5 * torch.mean(outputs - 1.0) ** 2
        #
        #     generated_outputs = generator(batch_noisy, batch_z)
        #     outputs = discriminator(torch.cat((generated_outputs, batch_noisy), dim=1), ref_batch2)
        #     noisy_loss = 0.5 * torch.mean(outputs ** 2)
        #
        #     d_loss = clean_loss + noisy_loss
        #     d_loss.backward()
        #     d_optimizer.step()
        #
        #     #  Generator的训练
        #     generator.zero_grad()
        #     generated_outputs = generator(batch_noisy, batch_z)
        #     gen_noisy_pair = torch.cat((generated_outputs, batch_noisy), dim=1)
        #     outputs = discriminator(gen_noisy_pair, ref_batch2)
        #
        #     g_loss = 0.5 * torch.mean((outputs - 1.0) ** 2)
        #     l1_dist = torch.abs(torch.add(generated_outputs, torch.neg(batch_clean)))
        #     g_cond_loss = 100 * torch.mean(l1_dist)
        #     g_loss = g_loss + g_cond_loss
        #
        #     g_loss.backward()
        #     g_optimizer.step()
        #
        #     print("Epoch %d:%d d clean loss %.4f, d noisy loss %.4f, g loss %.4f, g conditional loss %.4f" % (
        #         epoch + 1,
        #         i_batch,
        #         clean_loss,
        #         noisy_loss,
        #         g_loss,
        #         g_cond_loss))

        g_model_name = os.path.join(para.save_path, "G_" + str(epoch) + "_%.4f" % g_cond_loss + ".pkl")
        # d_model_name = os.path.join(para.save_path, "D_" + str(epoch) + "_%.4f" % noisy_loss + ".pkl")
        generator_state = {'model': generator.state_dict(), 'optimizer': g_optimizer.state_dict(), 'epoch': epoch}
        torch.save(generator_state, g_model_name)
        # discriminator_state = {'model': discriminator.state_dict(), 'optimizer': d_optimizer.state_dict(), 'epoch': epoch}
        # torch.save(discriminator_state, g_model_name)
