import torch
from dataset import SEGAN_Dataset, emphasis
from hparams import hparams
from model import Generator, Discriminator
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.autograd import Variable
#from adabelief_pytorch import AdaBelief
#from ranger21 import ranger21
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

    #  创建生成器
    generator = Generator()
    generator = generator.to(device)

    #  创建鉴别器
    discriminator = Discriminator()
    discriminator = discriminator.to(device)

    #  定义数据集
    m_dataset = SEGAN_Dataset(para)

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
    # g_optimizer = ranger21.Ranger21(generator.parameters(), lr=0.001, num_epochs=para.n_epoch,
    #                                 use_warmup=False,
    #                                 warmdown_active=False,
    #                                 num_batches_per_epoch=len(m_dataset.clean_files))
    # d_optimizer = ranger21.Ranger21(discriminator.parameters(), lr=0.001, num_epochs=para.n_epoch,
    #                                 use_warmup=False,
    #                                 warmdown_active=False,
    #                                 num_batches_per_epoch=len(m_dataset.clean_files))


    #  创建优化器radam
    g_optimizer = RAdam(generator.parameters(), lr=0.0001)
    d_optimizer = RAdam(discriminator.parameters(), lr=0.0001)

    #  获取ref_batch
    ref_batch = m_dataset.ref_batch(para.ref_batch_size)
    ref_batch = Variable(ref_batch)
    ref_batch = ref_batch.to(device)

    #  定义dataloader
    m_dataloader = DataLoader(m_dataset, batch_size=para.batch_size, shuffle=True, num_workers=0)
    loss_d_all = 0
    loss_g_all = 0
    n_step = 0



    for epoch in range(para.n_epoch):

        g_cond_loss = 0
        noisy_loss = 0
        for i_batch, sample_batch in enumerate(m_dataloader):
            batch_clean = sample_batch[0]
            batch_noisy = sample_batch[1]
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

            print("Epoch %d:%d d clean loss %.4f, d noisy loss %.4f, g loss %.4f, g conditional loss %.4f" % (epoch + 1,
                                                                                                              i_batch,
                                                                                                              clean_loss,
                                                                                                              noisy_loss,
                                                                                                              g_loss,
                                                                                                              g_cond_loss))

        g_model_name = os.path.join(para.save_path, "G_" + str(epoch) + "_%.4f" % g_cond_loss + ".pkl")
        d_model_name = os.path.join(para.save_path, "D_" + str(epoch) + "_%.4f" % noisy_loss + ".pkl")
        torch.save(generator.state_dict(), g_model_name)
        torch.save(discriminator.state_dict(), d_model_name)
