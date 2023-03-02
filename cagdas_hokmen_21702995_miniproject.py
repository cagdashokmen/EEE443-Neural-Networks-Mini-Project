import random
import sys
import numpy as np
import h5py as h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import pandas as pd
import seaborn as sn

question = sys.argv[1]


def cagdas_hokmen_21702995_hw1(question):
    if question == '1':
        with h5py.File('C:/users/ax24086/PycharmProjects/miniproject/data1.h5', 'r') as f1:
            img_renki = f1['data'][()]

        img_gri = 0.2126 * img_renki[:, 0, :, :] + 0.7152 * img_renki[:, 1, :, :] + 0.0722 * img_renki[
                                                                                                          :, 2, :, :]
        img_gri = img_gri - img_gri.mean(axis=(1, 2), keepdims=True)
        img_gri = np.clip(img_gri, -3 * img_gri.std(), 3 * img_gri.std())
        img_gri = (img_gri - img_gri.min()) / (img_gri.max() - img_gri.min())
        img_gri = img_gri * 0.8 + 0.1

        picsplot(img_renki, img_gri)

        first_attemp = Auto_Enc(256, 16, 0.05, 0.1, 0.5, 5e-4, 0.9)
        first_question_training(first_attemp, img_gri)
        showlayers(first_attemp, 'partc')

        second_attemp = Auto_Enc(256, 64, 0.05, 0.1, 0.5, 5e-4, 0.9)
        first_question_training(second_attemp, img_gri)
        showlayers(second_attemp, 'lyr-hid16')

        third_attemp = Auto_Enc(256, 100, 0.05, 0.1, 0.5, 5e-4, 0.9)
        first_question_training(third_attemp, img_gri)
        showlayers(third_attemp, 'lyr-hid100')

        fourth_attemp = Auto_Enc(256, 64, 0.05, 0.1, 0.5, 0, 0.9)
        first_question_training(fourth_attemp, img_gri)
        showlayers(fourth_attemp, 'lmbd-low')

        fifth_attemp = Auto_Enc(256, 64, 0.05, 0.1, 0.5, 10e-3, 0.9)
        first_question_training(fifth_attemp, img_gri)
        showlayers(fifth_attemp, 'lmbd-high')
        plt.show()
    elif question == '2':
        print(question)
        ##question 2 code goes here
        if __name__ == '__main__':
            with h5py.File('C:/users/ax24086/PycharmProjects/miniproject/data2.h5', 'r') as file:
                sample_training = file['trainx'][()] - 1
                lbl_train = file['traind'][()] - 1
                sample_val = file['valx'][()] - 1
                lbl_val = file['vald'][()] - 1
                sample_tst = file['testx'][()] - 1
                lbl_tst = file['testd'][()] - 1
                words_arr = file['words'][()].astype(str)

            netwrok1 = N_Network(8, [3 * 8, 64, 250])
            lbl_train_onehot = onehotfunc(lbl_train)
            sample_training_onehot = np.array([onehotfunc(sample_training[:, i]) for i in range(3)])
            sample_training_onehot = np.transpose(sample_training_onehot, axes=(1, 2, 0))

            lbl_val_onehot = onehotfunc(lbl_val)
            sample_val_onehot = np.array([onehotfunc(sample_val[:, i]) for i in range(3)])
            sample_val_onehot = np.transpose(sample_val_onehot, axes=(1, 2, 0))

            lbl_tst_onehot = onehotfunc(lbl_tst)
            sample_tst_onehot = np.array([onehotfunc(sample_tst[:, i]) for i in range(3)])
            sample_tst_onehot = np.transpose(sample_tst_onehot, axes=(1, 2, 0))

            netwrok1.S_G_D(zip(sample_training_onehot, lbl_train_onehot), zip(sample_val_onehot, lbl_val_onehot), 35,
                           200, 0.15, 0.85)
            with open("N_Network", 'wb') as out:
                pickle.dump(netwrok1, out, pickle.HIGHEST_PROTOCOL)
            print("test")
    elif question == '3':
        print(question)
        ##question 3 code goes here
        with h5py.File('C:/users/ax24086/PycharmProjects/miniproject/data3.h5', "r") as file:
            d_train = file['trX'][()]
            l_train = file['trY'][()].astype(int).argmax(axis=1)
            d_test = file['tstX'][()]
            l_test = file['tstY'][()].astype(int).argmax(axis=1)

        indxtrain = np.random.choice(np.arange(len(d_train)), int(0.9 * len(d_train)), replace=False)
        indxvali = list(set(range(len(d_train))) - set(indxtrain))

        d_validation = d_train[indxvali]
        l_validation = l_train[indxvali]
        d_train = d_train[indxtrain]
        l_train = l_train[indxtrain]

        minimumtr = d_train.min(axis=0)
        maximumtr = d_train.max(axis=0)
        d_train = (d_train - minimumtr) / (maximumtr - minimumtr)
        d_validation = (d_validation - minimumtr) / (maximumtr - minimumtr)
        d_test = (d_test - minimumtr) / (maximumtr - minimumtr)

        rcrnt = trainrc(d_train, l_train, d_validation, l_validation)
        rcrnttest(rcrnt, d_test, l_test)

        ntlstm = lstmtraining(d_train, l_train, d_validation, l_validation)
        ntlstmtest(ntlstm, d_test, l_test)

        ntgru = grutraining(d_train, l_train, d_validation, l_validation)
        ntgrutest(ntgru, d_test, l_test)

#### Classes and Functions ####
#### First Question ####
class Auto_Enc:
    def __init__(self, num_inp, num_lay, rate_lern, firstvalue, secondvalue, coefficient_reg, momentum):
        self.num_inp = num_inp
        self.num_lay = num_lay
        self.num_node = [num_lay, num_inp]
        self.rate_lern = rate_lern
        self.firstvalue = firstvalue
        self.secondvalue = secondvalue
        self.coefficient_reg = coefficient_reg
        self.momentum = momentum

        self.prevweight = {}
        self.prevbias = {}
        self.currentweight = {}
        self.currentbias = {}
        self.output_temporary = None
        self.active_temporary = None
        self.update()

    def prog_back(self, inp, labl):
        for x in reversed(range(0, len(self.num_node))):
            output = self.output_temporary[x]
            if x == 0:
                outputprev = inp
            else:
                outputprev = self.output_temporary[x - 1]
            act = self.active_temporary[x]
            if x == len(self.num_node) - 1:
                term_updated = (output - labl) * fonks_sigmder(act)
                loss_deg = self.function_loss(output, labl)
            else:
                outputmean = output.mean(axis=1)
                valu = (1 - self.firstvalue) / (1 - outputmean) - self.firstvalue / outputmean
                der_valu = self.secondvalue * valu.reshape(-1, 1)
                term_updated = (self.currentweight[f'w{x + 1}'].T @ term_updated + der_valu) * fonks_sigmder(
                    act)

            loss_derivative = self.coefficient_reg * self.currentweight[f'w{x}']
            chagedweight = np.matmul(term_updated, outputprev.T) / inp.shape[1] + loss_derivative
            changedbias = term_updated.mean(axis=1).reshape(-1, 1)
            updatedweight = self.rate_lern * chagedweight + self.momentum * self.prevweight[f'w{x}']
            updatedbias = self.rate_lern * changedbias + self.momentum * self.prevbias[f'b{x}']
            self.prevweight[f'w{x}'] = updatedweight
            self.prevbias[f'b{x}'] = updatedbias
            self.currentweight[f'w{x}'] -= updatedweight
            self.currentbias[f'b{x}'] -= updatedbias
        return loss_deg.mean()


    def outputfound(self, inpt):
        o = inpt
        self.output_temporary = []
        self.active_temporary = []
        for bs, wght in zip(self.currentbias.values(), self.currentweight.values()):
            activation = np.matmul(wght, o) + bs
            o = fonks_sigm(activation)
            self.output_temporary.append(o)
            self.active_temporary.append(activation)
        return o

    def update(self):
        for x, y in enumerate(self.num_node):
            if x == 0:
                number_layer_input = self.num_inp
            else:
                number_layer_input = self.num_node[x - 1]
            number_layer_output = y
            term_updated = np.sqrt(number_layer_input + number_layer_output)
            term_updated = np.sqrt(6) / term_updated
            wght = np.random.uniform(-term_updated, term_updated, (number_layer_output, number_layer_input))
            bs = np.random.uniform(-term_updated, term_updated, (number_layer_output, 1))
            self.currentweight[f'w{x}'] = wght
            self.currentbias[f'b{x}'] = bs
            self.prevweight[f'w{x}'] = np.zeros(wght.shape)
            self.prevbias[f'b{x}'] = np.zeros(bs.shape)



    def function_loss(self, inp, labl):
        menaserr = np.square(inp - labl).mean(axis=0)
        term_regul = np.sum([np.linalg.norm(w, 'fro') ** 2 for w in self.currentweight.values()])  # compute regularization term
        meanout = self.output_temporary[0].mean(axis=1)
        valu = self.firstvalue * np.log(self.firstvalue / meanout) + (1 - self.firstvalue) * np.log((1 - self.firstvalue) / (1 - meanout))
        valu = valu.sum()
        deg_cost = 0.5 * menaserr + self.coefficient_reg / 2 * term_regul + self.secondvalue * valu
        return deg_cost



def picsplot(img_clr, img_gray):
    rgb_sample = np.random.choice(range(img_clr.shape[0]), 200, replace=False)
    first_fig, axis_first_fig = plt.subplots(20, 10)
    second_figure, axis_second_fig = plt.subplots(20, 10)

    for fval, valu in enumerate(rgb_sample):
        rgbpic = (img_clr[valu] - img_clr[valu].min()) / (img_clr[valu].max() - img_clr[valu].min())
        graypic = (img_gray[valu] - img_gray[valu].min()) / (img_gray[valu].max() - img_gray[valu].min())
        x, y = np.unravel_index(fval, axis_first_fig.shape)
        axis_first_fig[x, y].imshow(rgbpic.T)
        axis_first_fig[x, y].axis('off')
        axis_second_fig[x, y].imshow(graypic.T, cmap='gray')
        axis_second_fig[x, y].axis('off')

def fonks_sigm(b):
    a = 1 / (1 + np.exp(-b))
    return a


def fonks_sigmder(c):
   a = fonks_sigm(c) * (1 - fonks_sigm(c))
   return a

def showlayers(fonk, outputmean):
    wght = fonk.currentweight['w0']
    outputmean = wght.shape[0]
    num_row = int(np.ceil(np.sqrt(outputmean)))
    num_column = num_row
    figure, axis_figure = plt.subplots(num_row, num_column)
    for xf, xg in enumerate(wght):
        xg = (xg - xg.min()) / (xg.max() - xg.min())
        hg, hh = np.unravel_index(xf, axis_figure.shape)
        axis_figure[hg, hh].imshow(xg.reshape(16, 16).T, cmap='gray')
        axis_figure[hg, hh].axis('off')

def first_question_training(fonk, da):
    lbls = [da[xr:xr + 32] for xr in range(0, len(da), 32)]
    batchesmini = [da[xr:xr + 23] for xr in range(0, len(da), 32)]
    versyon_sk = list(zip(batchesmini, lbls))
    lost = np.zeros(100)
    for xr in tqdm(range(100)):
        np.random.shuffle(versyon_sk)
        g = 0
        for imgage, outputmean in versyon_sk:
            imgage = imgage.reshape(imgage.shape[0], -1).T
            out = fonk.outputfound(imgage)
            g += fonk.prog_back(imgage, imgage)
        lost[xr] = g / len(versyon_sk)
    plt.figure()
    plt.plot(lost[0:xr], label='Loss From Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss From Cross-Entropy')
    plt.title('Loss From Cross-Entropy with respect to Epoch')
    plt.legend()





#### Second Question ####

class N_Network(object):

    def __init__(self, Dimesion, szs):
        self.Dimesion = Dimesion
        self.numberlayer = len(szs)
        self.szs = szs
        self.bs12 = [np.random.normal(loc=0, scale=0.01, size=(yr, 1)) for yr in szs[1:]]
        self.whgt12 = [np.random.normal(loc=0, scale=0.01, size=(yr, xr)) for xr, yr in zip(szs[:-1], szs[1:])]
        self.rndm = np.random.normal(0, 0.01, (Dimesion, 250))

    def Forw_Feed(self, arg):
        embededout = self.rndm @ arg
        arg = embededout.reshape(-1, 1)
        for inx, (bs, wght) in enumerate(zip(self.bs12, self.whgt12)):
            if inx == len(self.whgt12) - 1:
                arg = max_s(np.dot(wght, arg) + bs)
            else:
                arg = sigmoid(np.dot(wght, arg) + bs)
        return arg

    def S_G_D(self, datatr1, dataval1, epch, sizeminib, var1, var2):
        loss_validation = []
        accu_validation = []
        loss_training = []
        accu_training = []
        average_loss_training = []
        average_accu_training = []

        dataval1 = list(dataval1)
        datatr1 = list(datatr1)
        numbern = len(datatr1)

        previousembedded = np.zeros(self.rndm.shape)
        previousbias = [np.zeros(b.shape) for b in self.bs12]
        previouswght = [np.zeros(w.shape) for w in self.whgt12]

        for bb in tqdm(range(epch)):
            random.shuffle(datatr1)
            multiple_mini_batch = [datatr1[pdx:pdx + sizeminib] for pdx in range(0, numbern, sizeminib)]
            for mini in multiple_mini_batch:
                [previouswght, previousbias, previousembedded] = self.minibatch_upt(mini, var1, var2, previouswght,previousbias, previousembedded)
                [accu_training, loss_training] = self.evl_trn(mini, accu_training, loss_training)

            average_loss_training.append(np.mean(loss_training))
            average_accu_training.append(np.mean(accu_training))
            [accu_validation, loss_validation] = self.calculatedval(dataval1, accu_validation, loss_validation)
            if bb >= 2:
                if loss_validation[-1] >= loss_validation[-2]:
                    break

        plt.figure(1)
        plt.plot(loss_validation, linewidth=1.5, label='validation loss')
        plt.plot(average_loss_training, linewidth=1.5, label='training loss')
        plt.xlabel('epoch #')
        plt.legend()
        plt.grid()
        plt.ylim(0, 5)
        plt.tight_layout()
        plt.savefig(f"{'Q2a_loss'}.pdf", bbox_inches='tight', pad_inches=0.01)
        plt.figure(2)
        plt.plot(accu_validation, linewidth=1.5, label='validation accuracy')
        plt.plot(average_accu_training, linewidth=1.5, label='training accuracy')
        plt.xlabel('epoch #')
        plt.legend()
        plt.grid()
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"{'Q2_ACC'}.pdf", bbox_inches='tight', pad_inches=0.01)
        plt.show()
        with open("average_loss_training", 'wb') as outp2:
            pickle.dump(average_loss_training, outp2, pickle.HIGHEST_PROTOCOL)
        with open("average_accu_training", 'wb') as outp2:
            pickle.dump(average_accu_training, outp2, pickle.HIGHEST_PROTOCOL)
        with open("loss_validation", 'wb') as outp2:
            pickle.dump(loss_validation, outp2, pickle.HIGHEST_PROTOCOL)
        with open("accu_validation", 'wb') as outp2:
            pickle.dump(accu_validation, outp2, pickle.HIGHEST_PROTOCOL)

    def minibatch_upt(self, batchm, var1, var2, previouswght, previousbias, previousembedded):
        first_variable = [np.zeros(b.shape) for b in self.bs12]
        second_variable = [np.zeros(w.shape) for w in self.whgt12]
        embed_avg_var1 = np.zeros(self.rndm.shape)
        for rr, cc in batchm:
            dlt_first_variable, dlt_second_variable, embed_avg_var1_temp = self.propagation_backwards(rr, cc)
            first_variable = [inb + derb for inb, derb in zip(first_variable, dlt_first_variable)]
            second_variable = [nw + dnw for nw, dnw in zip(second_variable, dlt_second_variable)]
            embed_avg_var1 += embed_avg_var1_temp

        self.whgt12 = [wg - (var1 / len(batchm)) * wgt1 - var2 * wprev1 for wg, wgt1, wprev1 in zip(self.whgt12, second_variable, previouswght)]
        self.bs12 = [b1 - (var1 / len(batchm)) * bs1 - var2 * prevb1 for b1, bs1, prevb1 in zip(self.bs12, first_variable, previousbias)]
        previouswght = [(var1 / len(batchm)) * wgg + var2 * wgprev for wgg, wgprev in zip(second_variable, previouswght)]
        previousbias = [(var1 / len(batchm)) * bgg + var2 * bgprev for bgg, bgprev in zip(first_variable, previousbias)]

        previousembedded = var1 / len(batchm) * embed_avg_var1 + var2 * previousembedded
        self.rndm -= previousembedded
        return previouswght, previousbias, previousembedded

    def propagation_backwards(self, a, b):
        first_variable = [np.zeros(x.shape) for x in self.bs12]
        second_variable = [np.zeros(f.shape) for f in self.whgt12]
        active = self.rndm @ a
        active = active.reshape(-1, 1)

        actives = [active]
        st1 = []

        for i, (bs, wgt) in enumerate(zip(self.bs12, self.whgt12)):
            z = np.dot(wgt, active) + bs
            st1.append(z)
            active = max_s(z) if i == len(self.bs12) - 1 else sigmoid(z)
            actives.append(active)


        dlt = derivative_of_cost(st1[-1], b)
        first_variable[-1] = dlt
        second_variable[-1] = np.dot(dlt, actives[-2].transpose())

        for kl in range(2, self.numberlayer):
            m1 = st1[-kl]
            m1derv = derrsigmoid(m1)
            dlt = np.dot(self.whgt12[-kl + 1].transpose(), dlt) * m1derv
            first_variable[-kl] = dlt
            second_variable[-kl] = np.dot(dlt, actives[-kl - 1].transpose())


        dlt = (self.whgt12[0].T @ dlt)
        var3 = [dlt[self.Dimesion*i:self.Dimesion*(i+1)] @ a[:, i].reshape(1, -1) for i in range(3)]
        average_embedding_weight = np.mean(var3, axis=0)
        return first_variable, second_variable, average_embedding_weight

    def calculatedval(self, dataval1, accu_validation, loss_validation):
        resval = [(self.Forw_Feed(x), y) for (x, y) in dataval1]
        crosslossval = [entropcross(y, x) for (x, y) in resval]
        loss_validation.append(np.mean(crosslossval))
        accu_validation.append(sum(u.argmax() == og.argmax() for (u, og) in resval) / 46500)
        return accu_validation, loss_validation

    def evl_trn(self, batch_m, accu_training, loss_training):
        res_training = [(self.Forw_Feed(x), y) for (x, y) in batch_m]
        meansqloss = [entropcross(y, x) for (x, y) in res_training]
        accu_training.append((sum(x.argmax() == y.argmax() for (x, y) in res_training) / len(batch_m)))
        loss_training.append(np.mean(meansqloss))
        return accu_training, loss_training

def tanh(r):
    oyt = (np.exp(r) - np.exp(-r)) / (np.exp(r) + np.exp(-r))
    return oyt

def tanh_prime(r):
    a = 1 - tanh(r) * tanh(r)
    return a


def entropcross(a, b):
    min1 = -sum([a[i]*np.log(b[i]) for i in range(len(a))])
    return min1


def max_s(r):
    xop = np.exp(r-np.max(r))
    opp = xop / xop.sum(axis=0)
    return opp


def max_s_prime(r):
    b = max_s(r) * (1 - max_s(r))
    return b

def sigmoid(r):
    a = 1 / (1 + np.exp(-r))
    return a

def derrsigmoid(r):
    a = sigmoid(r) * (1 - sigmoid(r))
    return a


def derivative_of_cost(k, h):
    hotencod = h.argmax()
    op1 = max_s(k)
    op1[hotencod, 0] -= 1
    return op1


def onehotfunc(inp):
    op2 = np.zeros((len(inp), 250), dtype=np.bool_)
    op2[range(len(inp)), inp] = 1
    return op2



def second_question_loading(name):

    name = name
    f = h5py.File(name, "r")


    keycls = list(f.keys())[:]

    return [np.array(f[keycls[0]]), np.array(f[keycls[1]]), np.array(f[keycls[2]]), np.array(f[keycls[3]]),
            np.array(f[keycls[4]]), np.array(f[keycls[5]]), np.array(f[keycls[6]])]

#### QUESTION 3 ####
class Netw_Reccurrent:

    def __init__(self, hidden, num_feature, lengthbp, numoflist, rrlearn, momentum):
        self.hidden = hidden
        self.num_feature = num_feature
        self.lengthbp = lengthbp
        self.numoflist = numoflist
        self.sayi_in = hidden
        self.rrlearn = rrlearn
        self.momentum = momentum

        self.weightprevi = {}
        self.biasprevi = {}
        self.weightcurre = {}
        self.biascurre = {}
        self.temp_sta = None
        self.temp_op = None
        self.temp_reac = None
        self.temp_activeted = None
        self.intl()

    def intl(self):
        for k1, numara in enumerate(self.numoflist):
            lay_in_nu = self.sayi_in if k1 == 0 else self.numoflist[k1 - 1]
            lay_o_nu = numara
            xavierterm = np.sqrt(6 / (lay_in_nu + lay_o_nu))
            wght = np.random.uniform(-xavierterm, xavierterm, (lay_o_nu, lay_in_nu))
            bs = np.random.uniform(-xavierterm, xavierterm, (lay_o_nu, 1))
            self.weightcurre[f'w{k1}'] = wght
            self.biascurre[f'b{k1}'] = bs
            self.weightprevi[f'w{k1}'] = np.zeros(wght.shape)
            self.biasprevi[f'b{k1}'] = np.zeros(bs.shape)

        xavierterm = np.sqrt(6 / (self.hidden + self.hidden))
        self.weight_recersive = np.random.uniform(-xavierterm, xavierterm, (self.hidden, self.hidden))
        xavierterm = np.sqrt(6 / (self.hidden + self.num_feature))
        self.weightrec_inp = np.random.uniform(-xavierterm, xavierterm, (self.hidden, self.num_feature))
        self.biasreccer = np.zeros((self.hidden, 1))
        self.weightprevi['recs_w'] = np.zeros(self.weight_recersive.shape)
        self.weightprevi['rec_b'] = np.zeros(self.biasreccer.shape)
        self.weightprevi['recinp_w'] = np.zeros(self.weightrec_inp.shape)

    def outp_found(self, input1):  # function for forward pass
        self.temp_sta = np.zeros((input1.shape[0] + 1, self.hidden, input1.shape[2]))
        self.temp_reac = np.zeros((input1.shape[0], self.hidden, input1.shape[2]))
        self.temp_sta[0] = recursionold = np.zeros((self.hidden, input1.shape[2]))
        for ar in range(input1.shape[0]):
            input2 = input1[ar]
            actvition = self.weightrec_inp @ input2 + self.weight_recersive @ recursionold + self.biasreccer
            self.temp_reac[ar] = actvition
            recursionold = hypertan(actvition)
            self.temp_sta[ar + 1] = recursionold

        oput = recursionold
        self.temp_op = []
        self.temp_activeted = []
        for yu, (bs, wght) in enumerate(zip(self.biascurre.values(), self.weightcurre.values())):
            actvition = wght @ oput + bs
            self.temp_activeted.append(actvition.copy())
            oput = np.clip(ss_max(actvition), 1e-5, 1 - 1e-5) if yu == len(self.numoflist) - 1 else relu(actvition)
            self.temp_op.append(oput)

        return oput

    def p_bward(self, input1, label1):
        oput = self.temp_op[-1]
        for te in reversed(range(0, len(self.numoflist))):
            oput_previous = self.temp_sta[-1] if te == 0 else self.temp_op[te - 1]
            actvition = self.temp_activeted[te]
            if te == len(self.numoflist) - 1:
                updated_version = self.lossderact(actvition, label1)
                loss = self.loss(oput, label1)
            else:
                updated_version = (self.weightcurre[f'w{te + 1}'].T @ updated_version) * derivativerelu(actvition)
            wgt_delt = updated_version @ oput_previous.T / input1.shape[1]
            bs_delt = updated_version.mean(axis=1).reshape(-1, 1)
            self.weightprevi[f'w{te}'] = updated_version_w = self.rrlearn * wgt_delt + self.momentum * self.weightprevi[f'w{te}']
            self.biasprevi[f'b{te}'] = updated_version_b = self.rrlearn * bs_delt + self.momentum * self.biasprevi[f'b{te}']
            self.weightcurre[f'w{te}'] -= updated_version_w
            self.biascurre[f'b{te}'] -= updated_version_b

        updated_version = (self.weightcurre['w0'].T @ updated_version) * derivativehypertan(self.temp_reac[-1])
        recursiondeltaweg = np.zeros(self.weight_recersive.shape)
        recursioninpdeltaw = np.zeros(self.weightrec_inp.shape)
        recursiondeltabs = np.zeros(self.biasreccer.shape)
        for a in reversed(range(input1.shape[0] - self.lengthbp, input1.shape[0])):
            recursioninpdeltaw += updated_version @ input1[a].T
            recursiondeltaweg += updated_version @ self.temp_sta[a].T
            recursiondeltabs += updated_version.mean(axis=1).reshape(-1, 1)
            if a > 0:
                updated_version = (self.weight_recersive.T @ updated_version) * derivativehypertan(self.temp_reac[a - 1])
        self.weightprevi['recs_w'] = updated_version_recs = self.rrlearn * recursiondeltaweg / input1.shape[2] + self.momentum * self.weightprevi['recs_w']
        self.weightprevi['recinp_w'] = updated_version_recinp = self.rrlearn * recursioninpdeltaw / input1.shape[2] + self.momentum * self.weightprevi['recinp_w']
        self.weightprevi['rec_b'] = updated_version_recb = self.rrlearn * recursiondeltabs + self.momentum * self.weightprevi['rec_b']
        self.weight_recersive -= updated_version_recs
        self.weightrec_inp -= updated_version_recinp
        self.biasreccer -= updated_version_recb

        return loss.mean()

    def loss(self, ax, label1):
        o = -np.log(ax[label1, np.arange(len(label1))])
        return o

    def lossder(self, ax, label1):
        o = -1 / ax[label1, np.arange(len(label1))]
        return o

    def lossderact(self, aktif, label1):
        a = ss_max(aktif)
        a[label1, np.arange(len(label1))] -= 1
        return a


class LSTM:

    def __init__(self, hidden, num_feature, lengthbp, numoflist, rrlearn, momentum):
        self.hidden = hidden
        self.num_feature = num_feature
        self.lengthbp = lengthbp
        self.numoflist = numoflist
        self.sayi_in = hidden
        self.rrlearn = rrlearn
        self.momentum = momentum

        self.weightprevi = {}
        self.biasprevi = {}
        self.weightcurre = {}
        self.biascurre = {}
        self.statement_temporary_lstm = None
        self.temp_op = None
        self.temporarylstm_acitvated = None
        self.temp_activeted = None
        self.intl()

    def intl(self):
        for ix, numara in enumerate(self.numoflist):
            lay_in_nu = self.sayi_in if ix == 0 else self.numoflist[ix - 1]
            lay_o_nu = numara
            xavierterm = np.sqrt(6 / (lay_in_nu + lay_o_nu))
            wght = np.random.uniform(-xavierterm, xavierterm, (lay_o_nu, lay_in_nu))
            bias = np.random.uniform(-xavierterm, xavierterm, (lay_o_nu, 1))
            self.weightcurre[f'w{ix}'] = wght
            self.biascurre[f'b{ix}'] = bias
            self.weightprevi[f'w{ix}'] = np.zeros(wght.shape)
            self.biasprevi[f'b{ix}'] = np.zeros(bias.shape)

        xavierterm1 = np.sqrt(6 / (self.hidden + self.num_feature))
        xavierterm2 = np.sqrt(6 / (self.hidden + self.hidden))
        input_weight12 = np.random.uniform(-xavierterm1, xavierterm1, (self.hidden, self.num_feature))
        hidden_weight12 = np.random.uniform(-xavierterm2, xavierterm2, (self.hidden, self.hidden))
        biasover = np.ones((self.hidden, 1))
        self.wght_f = [input_weight12, hidden_weight12, biasover]

        input_weight12 = np.random.uniform(-xavierterm1, xavierterm1, (self.hidden, self.num_feature))
        hidden_weight12 = np.random.uniform(-xavierterm2, xavierterm2, (self.hidden, self.hidden))
        biasover = np.ones((self.hidden, 1))
        self.wght_i = [input_weight12, hidden_weight12, biasover]

        input_weight12 = np.random.uniform(-xavierterm1, xavierterm1, (self.hidden, self.num_feature))
        hidden_weight12 = np.random.uniform(-xavierterm2, xavierterm2, (self.hidden, self.hidden))
        biasover = np.ones((self.hidden, 1))
        self.wght_c = [input_weight12, hidden_weight12, biasover]

        input_weight12 = np.random.uniform(-xavierterm1, xavierterm1, (self.hidden, self.num_feature))
        hidden_weight12 = np.random.uniform(-xavierterm2, xavierterm2, (self.hidden, self.hidden))
        biasover = np.ones((self.hidden, 1))
        self.wght_o = [input_weight12, hidden_weight12, biasover]

        for l1 in ['f', 'i', 'c', 'o']:
            self.weightprevi[f'{l1}_w_inp'] = np.zeros((self.hidden, self.num_feature))
            self.weightprevi[f'{l1}_w_hid'] = np.zeros((self.hidden, self.hidden))
            self.weightprevi[f'{l1}_b_over'] = np.zeros((self.hidden, 1))

    def outp_found(self, input1):
        self.statement_temporary_lstm = np.zeros((6, input1.shape[0] + 1, self.hidden, input1.shape[2]))
        self.temporarylstm_acitvated = np.zeros((4, input1.shape[0], self.hidden, input1.shape[2]))
        self.statement_temporary_lstm[3, 0] = hucreprev = np.zeros((self.hidden, input1.shape[2]))
        self.statement_temporary_lstm[5, 0] = gprev = np.zeros((self.hidden, input1.shape[2]))
        for r in range(input1.shape[0]):
            input1_s = input1[r]
            aktif11 = self.wght_f[0] @ input1_s + self.wght_f[1] @ gprev + self.wght_f[2]
            self.temporarylstm_acitvated[0, r] = aktif11
            sinyal11 = sigmoid(aktif11)
            self.statement_temporary_lstm[0, r + 1] = sinyal11
            state11 = hucreprev * sinyal11

            aktifi = self.wght_i[0] @ input1_s + self.wght_i[1] @ gprev + self.wght_i[2]
            self.temporarylstm_acitvated[1, r] = aktifi
            sinyali = sigmoid(aktifi)
            self.statement_temporary_lstm[1, r + 1] = sinyali
            aktifcp = self.wght_c[0] @ input1_s + self.wght_c[1] @ gprev + self.wght_c[2]
            self.temporarylstm_acitvated[2, r] = aktif11
            sinyalcp = hypertan(aktifcp)
            self.statement_temporary_lstm[2, r + 1] = sinyalcp
            hucreprev = self.statement_temporary_lstm[3, r + 1] = state11 + (sinyalcp * sinyali)

            output_aactive = self.wght_o[0] @ input1_s + self.wght_o[1] @ gprev + self.wght_o[2]
            self.temporarylstm_acitvated[3, r] = output_aactive
            sinyalo = sigmoid(output_aactive)
            self.statement_temporary_lstm[4, r + 1] = sinyalo
            gprev = self.statement_temporary_lstm[5, r + 1] = sinyalo * hypertan(hucreprev)

        sinyalo = gprev
        self.temp_op = []
        self.temp_activeted = []
        for k, (bias, wght) in enumerate(zip(self.biascurre.values(), self.weightcurre.values())):
            aktif123 = wght @ sinyalo + bias
            self.temp_activeted.append(aktif123.copy())
            sinyalo = np.clip(ss_max(aktif123), 1e-5, 1 - 1e-5) if k == len(self.numoflist) - 1 else relu(aktif123)
            self.temp_op.append(sinyalo)

        return sinyalo

    def p_bward(self, input1, label1):
        out = self.temp_op[-1]
        for r in reversed(range(0, len(self.numoflist))):
            oput_previous = self.statement_temporary_lstm[-1, -1] if r == 0 else self.temp_op[r - 1]
            activ = self.temp_activeted[r]
            if r == len(self.numoflist) - 1:
                updated_version = self.lossderact(activ, label1)
                loss = self.loss(out, label1)
            else:
                updated_version = (self.weightcurre[f'w{r + 1}'].T @ updated_version) * derivativerelu(activ)  # compute update
            wgt_delt = updated_version @ oput_previous.T / input1.shape[1]
            bs_delt = updated_version.mean(axis=1).reshape(-1, 1)
            self.weightprevi[f'w{r}'] = updated_version_w = self.rrlearn * wgt_delt + self.momentum * self.weightprevi[f'w{r}']
            self.biasprevi[f'b{r}'] = updated_version_b = self.rrlearn * bs_delt + self.momentum * self.biasprevi[f'b{r}']
            self.weightcurre[f'w{r}'] -= updated_version_w
            self.biascurre[f'b{r}'] -= updated_version_b

        updated_version_t = self.weightcurre['w0'].T @ updated_version
        hypertanderc = derivativehypertan(self.statement_temporary_lstm[3, -1])
        hypertanc = hypertan((self.statement_temporary_lstm[3, -1]))
        inputnet = self.statement_temporary_lstm[1, -1]
        out = self.statement_temporary_lstm[4, -1]

        updated_version_labels = {'o': updated_version_t * hypertanc * derivativesigmoid(self.temporarylstm_acitvated[3, -1]),
                       'c': updated_version_t * out * hypertanderc * inputnet * derivativehypertan(self.temporarylstm_acitvated[2, -1]),
                       'i': updated_version_t * out * hypertanderc * self.statement_temporary_lstm[2, -1] * derivativesigmoid(self.temporarylstm_acitvated[1, -1]),
                       'f': updated_version_t * out * hypertanderc * self.statement_temporary_lstm[3, -2] * derivativesigmoid(self.temporarylstm_acitvated[0, -1])}

        updated_version_lstm = {}
        for l1 in ['f', 'i', 'c', 'o']:
            updated_version_lstm[f'{l1}_w_inp'] = np.zeros((self.hidden, self.num_feature))
            updated_version_lstm[f'{l1}_w_hid'] = np.zeros((self.hidden, self.hidden))
            updated_version_lstm[f'{l1}_b_over'] = np.zeros((self.hidden, 1))

        for k in reversed(range(input1.shape[0] - self.lengthbp, input1.shape[0])):
            for l1 in ['f', 'i', 'c', 'o']:
                updated_version_lstm[f'{l1}_w_inp'] += updated_version_labels[l1] @ input1[k].T / input1.shape[2]
                updated_version_lstm[f'{l1}_w_hid'] += updated_version_labels[l1] @ self.statement_temporary_lstm[-1, k].T / input1.shape[2]
                updated_version_lstm[f'{l1}_b_over'] += updated_version_labels[l1].mean(axis=1).reshape(-1, 1)

            if k <= 0:
                continue
            hypertanc = hypertan(self.statement_temporary_lstm[3, k])
            hypertanderc = derivativehypertan(self.statement_temporary_lstm[3, k])
            inputnet = self.statement_temporary_lstm[1, k]
            out = self.statement_temporary_lstm[4, k]

            updated_version_labels['o'] = (self.wght_o[1].T @ updated_version_labels['o']) * hypertanc * derivativesigmoid(self.temporarylstm_acitvated[3, k - 1])
            updated_version_labels['c'] = (self.wght_c[1].T @ updated_version_labels['c']) * hypertanderc * inputnet * derivativehypertan(self.temporarylstm_acitvated[2, k - 1])
            updated_version_labels['i'] = (self.wght_i[1].T @ updated_version_labels['i']) * out * hypertanderc * self.statement_temporary_lstm[2, k] * derivativesigmoid(self.temporarylstm_acitvated[1, k - 1])
            updated_version_labels['f'] = (self.wght_f[1].T @ updated_version_labels['f']) * out * hypertanderc * self.statement_temporary_lstm[3, k - 1] * derivativesigmoid(self.temporarylstm_acitvated[0, k - 1])

        for l1 in ['f', 'i', 'c', 'o']:
            updated_version_slstm = updated_version_lstm[f'{l1}_w_inp']
            self.weightprevi[f'{l1}_w_inp'] = dlt1 = self.rrlearn * updated_version_slstm + self.momentum * self.weightprevi[f'{l1}_w_inp']
            newweight = getattr(self, f'wght_{l1}')
            newweight[0] -= dlt1

            updated_version_slstm = updated_version_lstm[f'{l1}_w_hid']
            self.weightprevi[f'{l1}_w_hid'] = dlt1 = self.rrlearn * updated_version_slstm + self.momentum * self.weightprevi[f'{l1}_w_hid']
            newweight[1] -= dlt1

            updated_version_slstm = updated_version_lstm[f'{l1}_b_over']
            self.weightprevi[f'{l1}_b_over'] = dlt1 = self.rrlearn * updated_version_slstm + self.momentum * self.weightprevi[f'{l1}_b_over']
            newweight[2] -= dlt1

        return loss.mean()

    def loss(self, inx, label1):
        out = -np.log(inx[label1, np.arange(len(label1))])
        return out

    def lossder(self, inx, label1):
        out = -1 / inx[label1, np.arange(len(label1))]
        return out

    def lossderact(self, activ, label1):
        t = ss_max(activ)
        t[label1, np.arange(len(label1))] -= 1
        return t


class GRU:

    def __init__(self, hidden, num_feature, lengthbp, numoflist, rrlearn, momentum):
        self.hidden = hidden
        self.num_feature = num_feature
        self.lengthbp = lengthbp
        self.numoflist = numoflist
        self.sayi_in = hidden
        self.rrlearn = rrlearn
        self.momentum = momentum

        self.weightprevi = {}
        self.biasprevi = {}
        self.weightcurre = {}
        self.biascurre = {}
        self.temp_sta = None
        self.temp_op = None
        self.temp_reac = None
        self.temp_activeted = None
        self.intl()

    def intl(self):
        for k, numara in enumerate(self.numoflist):
            lay_in_nu = self.sayi_in if k == 0 else self.numoflist[k - 1]
            lay_o_nu = numara
            xavierterm = np.sqrt(6 / (lay_in_nu + lay_o_nu))
            weight = np.random.uniform(-xavierterm, xavierterm, (lay_o_nu, lay_in_nu))
            bias = np.random.uniform(-xavierterm, xavierterm, (lay_o_nu, 1))
            self.weightcurre[f'w{k}'] = weight
            self.biascurre[f'b{k}'] = bias
            self.weightprevi[f'w{k}'] = np.zeros(weight.shape)
            self.biasprevi[f'b{k}'] = np.zeros(bias.shape)

        xavierterm = np.sqrt(6 / (self.hidden + self.hidden))
        self.weight_recersive = np.random.uniform(-xavierterm, xavierterm, (self.hidden, self.hidden))
        xavierterm = np.sqrt(6 / (self.hidden + self.num_feature))
        self.weightrec_inp = np.random.uniform(-xavierterm, xavierterm, (self.hidden, self.num_feature))
        self.biasreccer = np.zeros((self.hidden, 1))
        self.weightprevi['recs_w'] = np.zeros(self.weight_recersive.shape)
        self.weightprevi['rec_b'] = np.zeros(self.biasreccer.shape)
        self.weightprevi['recinp_w'] = np.zeros(self.weightrec_inp.shape)

    def outp_found(self, input1):  # function for forward pass
        self.temp_sta = np.zeros((input1.shape[0] + 1, self.hidden, input1.shape[2]))
        self.temp_reac = np.zeros((input1.shape[0], self.hidden, input1.shape[2]))
        self.temp_sta[0] = recursionold = np.zeros((self.hidden, input1.shape[2]))
        for k in range(input1.shape[0]):
            input1_s = input1[k]
            activ = self.weightrec_inp @ input1_s + self.weight_recersive @ recursionold + self.biasreccer
            self.temp_reac[k] = activ
            recursionold = hypertan(activ)
            self.temp_sta[k + 1] = recursionold

        opp = recursionold
        self.temp_op = []
        self.temp_activeted = []
        for k, (bias, weight) in enumerate(zip(self.biascurre.values(), self.weightcurre.values())):
            activ = weight @ opp + bias
            self.temp_activeted.append(activ.copy())
            opp = np.clip(ss_max(activ), 1e-5, 1 - 1e-5) if k == len(self.numoflist) - 1 else relu(activ)
            self.temp_op.append(opp)

        return opp

    def p_bward(self, input1, label1):
        out = self.temp_op[-1]
        for k in reversed(range(0, len(self.numoflist))):
            oput_previous = self.temp_sta[-1] if k == 0 else self.temp_op[k - 1]
            activ = self.temp_activeted[k]
            if k == len(self.numoflist) - 1:
                updated_version = self.lossderact(activ, label1)
                loss = self.loss(out, label1)
            else:
                updated_version = (self.weightcurre[f'w{k + 1}'].T @ updated_version) * derivativerelu(activ)
            wgt_delt = updated_version @ oput_previous.T / input1.shape[1]
            bs_delt = updated_version.mean(axis=1).reshape(-1, 1)
            self.weightprevi[f'w{k}'] = updated_version_w = self.rrlearn * wgt_delt + self.momentum * self.weightprevi[f'w{k}']
            self.biasprevi[f'b{k}'] = updated_version_b = self.rrlearn * bs_delt + self.momentum * self.biasprevi[f'b{k}']
            self.weightcurre[f'w{k}'] -= updated_version_w
            self.biascurre[f'b{k}'] -= updated_version_b

        updated_version = (self.weightcurre['w0'].T @ updated_version) * derivativehypertan(self.temp_reac[-1])
        recursiondeltaweg = np.zeros(self.weight_recersive.shape)
        recursioninpdeltaw = np.zeros(self.weightrec_inp.shape)
        recursiondeltabs = np.zeros(self.biasreccer.shape)
        for k in reversed(range(input1.shape[0] - self.lengthbp, input1.shape[0])):
            recursioninpdeltaw += updated_version @ input1[k].T
            recursiondeltaweg += updated_version @ self.temp_sta[k].T
            recursiondeltabs += updated_version.mean(axis=1).reshape(-1, 1)
            if k > 0:
                updated_version = (self.weight_recersive.T @ updated_version) * derivativehypertan(self.temp_reac[k - 1])
        self.weightprevi['recs_w'] = updated_version_recs = self.rrlearn * recursiondeltaweg / input1.shape[2] + self.momentum * self.weightprevi['recs_w']
        self.weightprevi['recinp_w'] = updated_version_recinp = self.rrlearn * recursioninpdeltaw / input1.shape[2] + self.momentum * self.weightprevi['recinp_w']
        self.weightprevi['rec_b'] = updated_version_recb = self.rrlearn * recursiondeltabs + self.momentum * self.weightprevi['rec_b']
        self.weight_recersive -= updated_version_recs
        self.weightrec_inp -= updated_version_recinp
        self.biasreccer -= updated_version_recb

        return loss.mean()

    def loss(self, inx, label1):
        out = -np.log(inx[label1, np.arange(len(label1))])
        return out

    def lossder(self, inx, label1):
        out = -1 / inx[label1, np.arange(len(label1))]
        return out

    def lossderact(self, activ, label1):
        a = ss_max(activ)
        a[label1, np.arange(len(label1))] -= 1
        return a











def trainrc(dtraining, ltraning, d_validation, valid_label1):
    numberepoch = 50
    mmbatchsize = 32
    rcrnt = Netw_Reccurrent(128, 3, 15, [64, 16, 6], 0.0005, 0.85)
    minibatchevery = [dtraining[ind:ind + mmbatchsize] for ind in range(0, len(dtraining), mmbatchsize)]
    labelevery = [ltraning[ind:ind + mmbatchsize] for ind in range(0, len(ltraning), mmbatchsize)]
    ddtraining = list(zip(minibatchevery, labelevery))
    ddvalidation = list(zip(d_validation, valid_label1))

    lossfromtrainingg = np.zeros(numberepoch)
    accuracyfromtrainingg = np.zeros(numberepoch)
    lossfromvalidationn = np.zeros(numberepoch)
    accuracyfromvalidation = np.zeros(numberepoch)

    var11 = 0
    valuelossold = 10000
    for k in tqdm(range(numberepoch)):
        np.random.shuffle(ddtraining)
        loss = 0
        accuracy = 0
        lossvalu = 0
        accuvalu = 0
        trainconfig = np.zeros((6, 6))
        for data, label1 in ddtraining:
            data = np.transpose(data, (1, 2, 0))
            out = rcrnt.outp_found(data)
            accuracy += (out.argmax(axis=0) == label1).mean()
            loss += rcrnt.p_bward(data, label1)
            trainconfig[label1, out.argmax(axis=0)] += 1
        lossfromtrainingg[k] = loss / len(ddtraining)
        accuracyfromtrainingg[k] = (accuracy / len(ddtraining)) * 100

        for data, label1 in ddvalidation:
            label1 = np.array([label1])
            data = data.reshape(150, 3, 1)
            out = rcrnt.outp_found(data)
            accuvalu += (out.argmax(axis=0) == label1).mean()
            lossvalu += rcrnt.loss(out, label1).mean()
        lossfromvalidationn[k] = latest_valoss = lossvalu / len(ddvalidation)
        accuracyfromvalidation[k] = (accuvalu / len(ddvalidation)) * 100

        if latest_valoss < valuelossold:
            valuelossold = latest_valoss
            var11 = k
        elif k - var11 >= 5:
            break

    plt.figure()
    plt.plot(accuracyfromtrainingg[0:k], label='Training Accuracy')
    plt.plot(accuracyfromvalidation[0:k], label='Validation Accuracy')
    plt.legend()
    plt.xlabel('Epoch Num')
    plt.ylabel('Accuracy')
    plt.title('Percentage of Accuracy vs Epoch Num')
    plt.grid()

    plt.figure()
    plt.plot(lossfromtrainingg[0:k], label='Training Loss')
    plt.plot(lossfromvalidationn[0:k], label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch Number')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Cross-Entropy Loss vs Epoch Number')
    plt.grid()

    plt.show()

    trainconfig = trainconfig / 100.0
    matconfig = pd.DataFrame(trainconfig, index=['downstairs', 'jogging', 'sitting', 'standing', 'upstairs', 'walking'],
                              columns=['downstairs', 'jogging', 'sitting', 'standing', 'upstairs', 'walking'])
    sn.set(font_scale=1.4)
    plt.figure(figsize=(12, 6))
    sn.heatmap(matconfig, annot=True, annot_kws={"size": 16})
    plt.xlabel('inxicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=0)
    plt.show()

    return rcrnt


def rcrnttest(net, d_test, l_test):
    aacuraccytest = 0
    conftesting = np.zeros((6, 6))
    for data, label1 in zip(d_test, l_test):
        label1 = np.array([label1])
        data = data.reshape(150, 3, 1)
        out = net.outp_found(data)
        conftesting[label1, out.argmax(axis=0)] += 1
        aacuraccytest += (out.argmax(axis=0) == label1).mean()
    aacuraccytest = aacuraccytest / len(l_test)

    print(aacuraccytest)

    conftesting = conftesting / 100.0
    matconfig = pd.DataFrame(conftesting, index=['downstairs', 'jogging', 'sitting', 'standing', 'upstairs', 'walking'],
                              columns=['downstairs', 'jogging', 'sitting', 'standing', 'upstairs', 'walking'])
    sn.set(font_scale=1.4)
    plt.figure(figsize=(12, 6))
    sn.heatmap(matconfig, annot=True, annot_kws={"size": 16})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=0)
    plt.show()


def lstmtraining(dtraining, ltraning, d_validation, valid_label1):
    numberepoch = 50
    mmbatchsize = 32
    ntlstm = LSTM(128, 3, 50, [64, 16, 6], 0.0005, 0.85)
    minibatchevery = [dtraining[ind:ind + mmbatchsize] for ind in range(0, len(dtraining), mmbatchsize)]
    labelevery = [ltraning[ind:ind + mmbatchsize] for ind in range(0, len(ltraning), mmbatchsize)]
    ddtraining = list(zip(minibatchevery, labelevery))
    ddvalidation = list(zip(d_validation, valid_label1))

    lossfromtrainingg = np.zeros(numberepoch)
    accuracyfromtrainingg = np.zeros(numberepoch)
    lossfromvalidationn = np.zeros(numberepoch)
    accuracyfromvalidation = np.zeros(numberepoch)

    var11 = 0
    valuelossold = 10000
    for k in tqdm(range(numberepoch)):
        np.random.shuffle(ddtraining)
        loss = 0
        acc = 0
        lossvalu = 0
        accuvalu = 0
        trainconfig = np.zeros((6, 6))
        for data, label1 in ddtraining:
            data = np.transpose(data, (1, 2, 0))
            out = ntlstm.outp_found(data)
            acc += (out.argmax(axis=0) == label1).mean()
            loss += ntlstm.p_bward(data, label1)
            trainconfig[label1, out.argmax(axis=0)] += 1
        lossfromtrainingg[k] = loss / len(ddtraining)
        accuracyfromtrainingg[k] = (acc / len(ddtraining)) * 100

        for data, label1 in ddvalidation:
            label1 = np.array([label1])
            data = data.reshape(150, 3, 1)
            out = ntlstm.outp_found(data)
            accuvalu += (out.argmax(axis=0) == label1).mean()
            lossvalu += ntlstm.loss(out, label1).mean()
        lossfromvalidationn[k] = latest_valoss = lossvalu / len(ddvalidation)
        accuracyfromvalidation[k] = (accuvalu / len(ddvalidation)) * 100

        if latest_valoss < valuelossold:
            valuelossold = latest_valoss
            var11 = k
        elif k - var11 >= 7:
            break

    plt.figure()
    plt.plot(accuracyfromtrainingg[0:k], label='Training Accuracy')
    plt.plot(accuracyfromvalidation[0:k], label='Validation Accuracy')
    plt.legend()
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.title('Percentage Accuracy vs Epoch Number')
    plt.grid()

    plt.figure()
    plt.plot(lossfromtrainingg[0:k], label='Training Loss')
    plt.plot(lossfromvalidationn[0:k], label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch Number')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Cross-Entropy Loss vs Epoch Number')
    plt.grid()
    plt.show()

    trainconfig = trainconfig / 100.0
    matconfig = pd.DataFrame(trainconfig, index=['downstairs', 'jogging', 'sitting', 'standing', 'upstairs', 'walking'],
                              columns=['downstairs', 'jogging', 'sitting', 'standing', 'upstairs', 'walking'])
    sn.set(font_scale=1.4)
    plt.figure(figsize=(12, 6))
    sn.heatmap(matconfig, annot=True, annot_kws={"size": 16})  # font size
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=0)
    plt.show()

    return ntlstm


def ntlstmtest(net, d_test, l_test):
    aacuraccytest = 0
    conftesting = np.zeros((6, 6))
    for data, label1 in zip(d_test, l_test):
        label1 = np.array([label1])
        data = data.reshape(150, 3, 1)
        out = net.outp_found(data)
        conftesting[label1, out.argmax(axis=0)] += 1
        aacuraccytest += (out.argmax(axis=0) == label1).mean()
    aacuraccytest = aacuraccytest / len(l_test)

    print(aacuraccytest)

    conftesting = conftesting / 100.0
    matconfig = pd.DataFrame(conftesting, index=['downstairs', 'jogging', 'sitting', 'standing', 'upstairs', 'walking'],
                              columns=['downstairs', 'jogging', 'sitting', 'standing', 'upstairs', 'walking'])
    sn.set(font_scale=1.4)
    plt.figure(figsize=(12, 6))
    sn.heatmap(matconfig, annot=True, annot_kws={"size": 16})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=0)
    plt.show()


def grutraining(dtraining, ltraning, d_validation, valid_label1):
    numberepoch = 50
    mmbatchsize = 32
    ntgru = GRU(128, 3, 15, [64, 32, 6], 0.0005, 0.85)
    minibatchevery = [dtraining[ind:ind + mmbatchsize] for ind in range(0, len(dtraining), mmbatchsize)]
    labelevery = [ltraning[ind:ind + mmbatchsize] for ind in range(0, len(ltraning), mmbatchsize)]
    ddtraining = list(zip(minibatchevery, labelevery))
    ddvalidation = list(zip(d_validation, valid_label1))

    lossfromtrainingg = np.zeros(numberepoch)
    accuracyfromtrainingg = np.zeros(numberepoch)
    lossfromvalidationn = np.zeros(numberepoch)
    accuracyfromvalidation = np.zeros(numberepoch)

    var11 = 0
    valuelossold = 10000
    for k in tqdm(range(numberepoch)):
        np.random.shuffle(ddtraining)
        loss = 0
        acc = 0
        lossvalu = 0
        accuvalu = 0
        trainconfig = np.zeros((6, 6))
        for data, label1 in ddtraining:
            data = np.transpose(data, (1, 2, 0))
            out = ntgru.outp_found(data)
            acc += (out.argmax(axis=0) == label1).mean()
            loss += ntgru.p_bward(data, label1)
            trainconfig[label1, out.argmax(axis=0)] += 1
        lossfromtrainingg[k] = loss / len(ddtraining)
        accuracyfromtrainingg[k] = (acc / len(ddtraining)) * 100

        for data, label1 in ddvalidation:
            label1 = np.array([label1])
            data = data.reshape(150, 3, 1)
            out = ntgru.outp_found(data)
            accuvalu += (out.argmax(axis=0) == label1).mean()
            lossvalu += ntgru.loss(out, label1).mean()
        lossfromvalidationn[k] = latest_valoss = lossvalu / len(ddvalidation)
        accuracyfromvalidation[k] = (accuvalu / len(ddvalidation)) * 100

        if latest_valoss < valuelossold:
            valuelossold = latest_valoss
            var11 = k
        elif k - var11 >= 5:
            break

    plt.figure()
    plt.plot(accuracyfromtrainingg[0:k], label='Training Accuracy')
    plt.plot(accuracyfromvalidation[0:k], label='Validation Accuracy')
    plt.legend()
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.title('Percentage Accuracy vs Epoch Number')
    plt.grid()

    plt.figure()
    plt.plot(lossfromtrainingg[0:k], label='Training Loss')
    plt.plot(lossfromvalidationn[0:k], label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch Number')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Cross-Entropy Loss vs Epoch Number')
    plt.grid()
    plt.show()

    trainconfig = trainconfig / 100.0
    matconfig = pd.DataFrame(trainconfig, index=['downstairs', 'jogging', 'sitting', 'standing', 'upstairs', 'walking'],
                              columns=['downstairs', 'jogging', 'sitting', 'standing', 'upstairs', 'walking'])
    sn.set(font_scale=1.4)
    plt.figure(figsize=(12, 6))
    sn.heatmap(matconfig, annot=True, annot_kws={"size": 16})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=0)
    plt.show()

    return ntgru


def ntgrutest(net, d_test, l_test):
    aacuraccytest = 0
    conftesting = np.zeros((6, 6))
    for data, label1 in zip(d_test, l_test):
        label1 = np.array([label1])
        data = data.reshape(150, 3, 1)
        out = net.outp_found(data)
        conftesting[label1, out.argmax(axis=0)] += 1
        aacuraccytest += (out.argmax(axis=0) == label1).mean()
    aacuraccytest = aacuraccytest / len(l_test)

    print(aacuraccytest)

    conftesting = conftesting / 100.0
    matconfig = pd.DataFrame(conftesting, index=['downstairs', 'jogging', 'sitting', 'standing', 'upstairs', 'walking'],
                              columns=['downstairs', 'jogging', 'sitting', 'standing', 'upstairs', 'walking'])
    sn.set(font_scale=1.4)
    plt.figure(figsize=(12, 6))
    sn.heatmap(matconfig, annot=True, annot_kws={"size": 16})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=0)
    plt.show()


def hypertan(xin):
    out = np.tanh(xin)
    return out


def derivativehypertan(xin):
    out = 1 / (np.cosh(xin) ** 2)
    return out


def ss_max(xin):
    expo = np.exp(xin - np.max(xin, axis=0))
    out = expo / np.sum(expo, axis=0)
    return out


def sigmoid(xin):
    out = 1 / (1 + np.exp(-xin))
    return out


def derivativesigmoid(xin):
    out = sigmoid(xin) * (1 - sigmoid(xin))
    return out


def relu(xin):
    xin[xin < 0] = 0
    return xin


def derivativerelu(xin):
    xin[xin < 0] = 0
    xin[xin >= 0] = 1
    return xin




cagdas_hokmen_21702995_hw1(question)
