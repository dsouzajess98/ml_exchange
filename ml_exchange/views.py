from django.shortcuts import render, redirect
from ml_exchange.models import *
from .models import *
import ipfsApi
import pandas as pd
from django.contrib.auth import logout
from dutils.dataset import DemoDataset, SampleCircleDataset, SampleAcrossCornerDataset
from dutils.dataset import SampleHalfDividedDataset
from dutils.neural_network import NeuralNetwork
from dutils.classification import Classification
from dutils.classification_2 import Classification2
import dutils.debug as dbg
from secrets import randbelow
from populus.utils.wait import wait_for_transaction_receipt
from django.core.files import File
from web3 import Web3
from populus import Project
from celery.decorators import task
from django.contrib.auth.models import User
import joblib
from populus.wait import Wait
import time
import _thread
from matplotlib import pyplot as plt
import numpy as np
import xlsxwriter
from django.conf import settings
import json
import random
global ml_exchange, scd, w_scale, b_scale, web3, timeout, total_gas_used, pending_tasks,chain
from random import seed
from random import random
import joblib
from sklearn.metrics import classification_report,confusion_matrix


def init():
    if settings.CHECK == 0:
        project = Project()
        chain_name = "tester"
        print("Make sure {} chain is running, you can connect to it, or you'll get timeout".format(chain_name))
        check = 0
        global chain
        with project.get_chain(chain_name) as chain:
            global ml_exchange, scd, w_scale, b_scale, web3, timeout, total_gas_used, pending_tasks, training_dataset, submission_tasks, ether_costs, acc
            acc =0
            web3 = chain.web3
            training_dataset = []
            accuracy_criteria = 5000  # 50.00%
            total_gas_used = 0
            timeout = 180
            w_scale = 1000  # Scale up weights by 1000x
            b_scale = 1000  # Scale up biases by 1000x
            pending_tasks = []
            submission_tasks = []

            ml_exchange, deploy_tx_hash = chain.provider.get_or_deploy_contract('ml_exchange_test')
            if deploy_tx_hash is not None:
                fund_receipt = wait_for_transaction_receipt(web3, deploy_tx_hash, timeout=timeout)
                total_gas_used += fund_receipt["gasUsed"]
                print("Fund gas: " + str(fund_receipt["gasUsed"]))
        scd = DemoDataset(training_percentage=0.8, partition_size=25)
        settings.CHECK = 1


init()


def scale_packed_data(data, scale):
    # Scale data and convert it to an integer
    return list(map(lambda x: int(x*scale), data))


def binary_2_one_hot(data):
    # Convert binary class data for one-hot training
    # TODO: Make this work for higher-dimension data
    rVal = []
    for data_point in data:
        new_dp = []
        input_data = data_point[:2]
        output_class = data_point[2:][0]
        if output_class == 0:
            output_data = [1,0]
        elif output_class == 1:
            output_data = [0,1]
        else:
            raise Exception("Data should only have 2 classes.")
        new_dp.extend(input_data)
        new_dp.extend(output_data)
        rVal.append(tuple(new_dp))
    return rVal


@task(name="submission")
def create_submission(user, id):
    global ml_exchange, scd, w_scale, b_scale, web3, timeout, total_gas_used, pending_tasks, chain
    print(ml_exchange.call().get_train_data_length(id - 1))
    # Get the training data from the contract
    contract_train_data = []
    contract_train_data_length = ml_exchange.call().get_train_data_length(id-1)
    for i in range(contract_train_data_length):
        for j in range(scd.dps):
            contract_train_data.append(ml_exchange.call().train_data(id-1, i, j))
    contract_train_data = scd.unpack_data(contract_train_data)
    print("Contract training data: " + str(contract_train_data))

    import csv
    csvfile = open('/home/jessica/ml_exchange_project/data/test_data.csv', 'w', newline='\n')
    obj = csv.writer(csvfile)
    for data in contract_train_data:
        obj.writerow(data)
    csvfile.close()

    submission = Submission()
    submission.contract = SmartContract.objects.get(id=id)
    submission.user = User.objects.get(id=user)
    submission.save()

    import webbrowser
    webbrowser.open('/home/jessica/ml_exchange_project/data/test_data.csv')

    return


@task(name="run_model")
def run_model(*argv):
    contract_train_data = argv[1]

    global scd
    contract_train_data = scd.unpack_data(contract_train_data)
    # contract_train_data = binary_2_one_hot(contract_train_data)
    print(contract_train_data)
    if argv[0] == 0:
        classification = Classification(contract_train_data, 2)
    else:
        classification = Classification2(contract_train_data, 2)
    return classification.weights_and_biases()


@task(name="model")
def submit_model(user, id):
    global ml_exchange, scd, w_scale, b_scale, web3, timeout, total_gas_used, pending_tasks, chain, submission_tasks
    total_gas_used = 0
    # offer_account = pseudo.objects.get(user=user)
    solver_account = web3.eth.accounts[user]

    if len(training_dataset) == 0:

        contract_train_data = []
        contract_train_data_length = ml_exchange.call().get_train_data_length(id - 1)
        for i in range(contract_train_data_length):
            for j in range(scd.dps):
                contract_train_data.append(ml_exchange.call().train_data(id - 1, i, j))

    else:
        contract_train_data = training_dataset
    # contract_train_data = scd.unpack_data(contract_train_data)

    print(contract_train_data)
    submission = Submission.objects.filter(contract=id, user=user)
    if submission[0].model == 'QmUX5HznL9grUn4wkUQFYyAAR9uzQ7jh8D4eQx1QgQcHoh':
        il_nn = 2
        hl_nn = [2, 2]
        ol_nn = 1
        task = run_model.apply_async(queue='low', args=(0, contract_train_data))
    else:
        il_nn = 2
        hl_nn = [4, 4, 4]
        ol_nn = 1
        task = run_model.apply_async(queue='low', args=(1, contract_train_data))
    # total_gas_used = 0
    # # Train a neural network with contract data
    # nn = NeuralNetwork(il_nn, hl_nn, ol_nn, 0.001, 1000000, 5, 100000)
    # contract_train_data = nn.binary_2_one_hot(contract_train_data)
    # nn.load_train_data(contract_train_data)
    # nn.init_network()
    # nn.train()

    while task.state != 'SUCCESS':
        continue
    nn = task.result
    nn = json.loads(nn)
    # trained_weights = nn.weights
    # trained_biases = nn.bias
    #
    # print("Trained weights: " + str(trained_weights))
    # print("Trained biases: " + str(trained_biases))
    # return
    # packed_trained_weights = nn.pack_weights(trained_weights)
    # print("Packed weights: " + str(packed_trained_weights))
    #
    # packed_trained_biases = nn.pack_biases(trained_biases)
    # print("Packed biases: " + str(packed_trained_biases))
    #
    # int_packed_trained_weights = scale_packed_data(packed_trained_weights, \
    #                                                w_scale)
    # print("Packed integer weights: " + str(int_packed_trained_weights))
    #
    # int_packed_trained_biases = scale_packed_data(packed_trained_biases, \
    #                                               b_scale)
    # print("Packed integer biases: " + str(int_packed_trained_biases))

    print("Solver address: " + str(User.objects.get(id=user).username))

    # Submit the solution to the contract
    submit_tx = ml_exchange.transact().submit_model(solver_account, il_nn, ol_nn, hl_nn, \
                                                                           nn["weights"],
                                                                           nn["biases"], id-1)
    submission_tasks.append(submit_tx)
    submit_receipt = wait_for_transaction_receipt(web3, submit_tx, timeout=timeout)
    total_gas_used += submit_receipt["gasUsed"]
    print("Submit gas: " + str(submit_receipt["gasUsed"]))
    chain.wait.for_receipt(submit_tx)

    # Get submission index ID
    submission_id = ml_exchange.call().get_submission_id(solver_account, il_nn, \
                                                         ol_nn, hl_nn, nn["weights"], nn["biases"], id-1)
    submission = Submission.objects.filter(contract=id, user=user)
    s = Submission.objects.get(id=submission[0].id)
    s.accuracy = nn["accuracy"]
    global acc
    acc =acc + nn["accuracy"]
    s.save()
    print("Submission ID: " + str(submission_id))
    return


@task(name="test_contract")
def test_contracts(user, id):
    print("Hello ready to test")
    global ml_exchange, scd, w_scale, b_scale, web3, timeout, total_gas_used, pending_tasks, chain, acc
    contract = SmartContract.objects.get(id=id)
    contract.testing_phase = True
    contract.submission_phase = False
    contract.save()

    # Wait until the submission period ends
    submission_t = ml_exchange.call().submission_stage_block_size(id - 1)  # get submission timeframe
    evaluation_t = ml_exchange.call().evaluation_stage_block_size(id-1)  # get evaluation timeframe
    test_reveal_t = ml_exchange.call().reveal_test_data_groups_block_size(id-1)  # get revealing testing dataset timeframe
    print(submission_t)
    submission_t = 40
    evaluation_t = 2
    test_reveal_t = 3
    # chain.wait.for_block(20 + submission_t)
    # time.sleep(60)
    for x in pending_tasks:
        while not x.ready():
            continue
    print("Hello ready to test")

    total_gas_used = 0
    testing_partition = list(map(lambda x: ml_exchange.call().testing_partition(id - 1, x), range(scd.num_test_data_groups)))
    offer_account = web3.eth.accounts[user]
    # solver_account = web3.eth.accounts[1]

    train_data = scd.pack_data(scd.train_data)
    test_data = scd.pack_data(scd.test_data)

    # Reveal the testing dataset after the submission period ends
    reveal_tx = []
    for i in range(len(testing_partition)):
        start = i * scd.dps * scd.partition_size
        end = start + scd.dps * scd.partition_size
        print("(" + str(testing_partition[i]) + ") Test data,nonce: " + str(test_data[start:end]) + "," + str(
            scd.test_nonce[i]))
        iter_tx = ml_exchange.transact({'from': offer_account}).reveal_test_data(test_data[start:end],
                                                                                 scd.test_nonce[i], id-1)
        iter_receipt = wait_for_transaction_receipt(web3, iter_tx, timeout=timeout)
        total_gas_used += iter_receipt["gasUsed"]
        print("Reveal test data iter " + str(i) + " gas: " + str(iter_receipt["gasUsed"]))
        reveal_tx.append(iter_tx)
        chain.wait.for_receipt(reveal_tx[i])

    contract_test_data = []
    contract_test_data_length = ml_exchange.call().get_test_data_length(id - 1)
    for i in range(contract_test_data_length):
        for j in range(scd.dps):
            contract_test_data.append(ml_exchange.call().test_data(id - 1, i, j))
    contract_test_data = scd.unpack_data(contract_test_data)
    diabetes = pd.DataFrame(contract_test_data)
    print(diabetes.head())

    y = diabetes.iloc[:, 2]
    X = diabetes.iloc[:, :2]

    submission = Submission.objects.filter(contract=id)
    i = 0
    best = 0
    ether = 0
    for s in submission:
        if s.model == 'QmUX5HznL9grUn4wkUQFYyAAR9uzQ7jh8D4eQx1QgQcHoh':
            loaded_model = joblib.load("finalized_model.sav")
            print("hello123")
        else:
            print("hello345")
            loaded_model = joblib.load("finalized_model_2.sav")
            # Evaluate the submitted solution
        score = loaded_model.score(X, y)

        print("Submission id: ", i, " Score: ", score)
        if best < score:
            predict_test = loaded_model.predict(X)
            from sklearn.metrics import classification_report, confusion_matrix
            report = classification_report(y, predict_test, output_dict = True)
            macro_precision = report['macro avg']['precision']
            macro_recall = report['macro avg']['recall']
            ether = 0.5 * score + 0.25 * macro_recall + 0.25 + macro_precision
            best = score
        eval_tx = ml_exchange.transact({'from': offer_account}).evaluate_model(i, id - 1,
                                                                               int(score * 10000))
        eval_receipt = wait_for_transaction_receipt(web3, eval_tx, timeout=timeout)
        total_gas_used += eval_receipt["gasUsed"]
        print("Eval gas: " + str(eval_receipt["gasUsed"]))
        i = i + 1

    # Wait until the test reveal period ends
    # chain.wait.for_block(20 + submission_t + test_reveal_t)



    # Wait until the evaluation period ends
    # chain.wait.for_block(20 + submission_t + test_reveal_t + evaluation_t)

    bal2 = web3.eth.getBalance(offer_account)

    # Finalize the contract
    final_tx = ml_exchange.transact({'from': offer_account}).finalize_contract(id-1)
    final_receipt = wait_for_transaction_receipt(web3, final_tx, timeout=timeout)
    total_gas_used += final_receipt["gasUsed"]
    print("Final gas: " + str(final_receipt["gasUsed"]))
    contract = SmartContract.objects.get(id=id)
    contract_finalized = ml_exchange.call().contract_terminated(id-1)
    contract.contract_active = False
    contract.testing_phase = False
    print("Contract finalized: " + str(contract_finalized))

    # Get best submission accuracy & ID
    best_submission_accuracy = ml_exchange.call().best_submission_accuracy(id-1)/10000
    best_submission_index = ml_exchange.call().best_submission_index(id - 1)

    print("Best submission ID: " + str(best_submission_index))
    print("Best submission accuracy: " + str(best_submission_accuracy))
    contract.best_submission = best_submission_accuracy
    contract.best_submission_user = submission[best_submission_index].user
    contract.save()

    print("Transferring ether: ", ether*10)
    # il_nn = 2
    # hl_nn = [4, 4]
    # ol_nn = 1
    #
    # l_nn = [il_nn] + hl_nn + [ol_nn]
    # input_layer = train_data[:2]
    # hidden_layers = [0] * sum(hl_nn)
    # output_layer = [0] * ol_nn
    # weights = ml_exchange.call().get_trained_weights(id-1, 0)
    # biases = ml_exchange.call().get_trained_biases(id-1, 0)
    # # Test forward
    # fwd_pass2 = ml_exchange.call().forward_pass2(l_nn, input_layer, hidden_layers, output_layer, weights,
    #                                              biases)
    #
    # nn = NeuralNetwork(il_nn, hl_nn, ol_nn, 0.001, 1000000, 5, 100000)
    # print("Test input: " + str(train_data[:2]))
    # print("Expected output: " + str(train_data[2]))
    # print("local nn prediction: " + str(nn.predict([train_data[:2]])))
    #
    # print("forward_pass2: " + str(fwd_pass2))

    print("Total gas used: " + str(total_gas_used))

    return


def upload_model(request):
    global ml_exchange, scd, w_scale, b_scale, web3, timeout, total_gas_used, pending_tasks, chain
    submission = Submission.objects.filter(user = request.user, contract = int(request.POST['id']))
    s = Submission.objects.get(id=submission[0].id)
    s.model = request.FILES['model']
    s.save()
    x = submit_model.apply_async(queue='default', kwargs={'user':request.user.id, 'id':int(request.POST['id']) })
    pending_tasks.append(x)
    return redirect('/accounts/profile/')


def get_training_data(request, id):
    global ml_exchange, scd, w_scale, b_scale, web3, timeout, total_gas_used, pending_tasks, chain
    submission  = Submission.objects.filter(contract=id,user=request.user)
    if len(submission) == 0:
        create_submission.apply_async(queue='default', args=(request.user.id, id))

    return redirect('/accounts/profile/')


def new_request(request, type):
    req = Request()
    req.from_user = request.user
    contract = SmartContract.objects.get(id=int(request.POST['id']))
    req.to_user = contract.user
    req.contract = contract
    if type == 0:
        dataset = None
        try:
            dataset = request.FILES['dataset']
        except:
            pass
        if dataset is None:
            req.request_type = 0
        else:
            req.request_type = 1
            req.dataset = dataset
            # update dataset
    elif type == 1:
        req.request_type = 2
        try:
            req.dataset = request.FILES['dataset']
            req.document = request.FILES['document']
        except:
            pass
    elif type == 2:
        req.to_user = contract.best_submission_user
    req.save()

    return redirect('/accounts/profile/')


@task(name="collab_with_data")
def collab_with_data(id, type):
    global ether_costs
    submission = Submission.objects.filter(contract=id)
    req = Request.objects.get(id=type)
    accuracy = 0
    res = []
    if req.dataset == 'QmUX5HznL9grUn4wkUQFYyAAR9uzQ7jh8D4eQx1QgQcHoh':
        d = pd.read_csv('/home/jessica/ml_exchange_project/data/good_data.csv', sep=',', header=0)
    else:
        d = pd.read_csv('/home/jessica/ml_exchange_project/data/bad_data.csv', sep=',', header=0)
    for s in submission:
        if s.model == 'QmUX5HznL9grUn4wkUQFYyAAR9uzQ7jh8D4eQx1QgQcHoh':
            loaded_model = joblib.load("finalized_model.sav")
        else:
            loaded_model = joblib.load("finalized_model_2.sav")

        y = d.iloc[:, 2]
        X = d.iloc[:, :2]

        loaded_model.fit(X,y)
        predict_train = loaded_model.predict(X)
        accuracy +=loaded_model.score(X,y)

        print(confusion_matrix(y, predict_train))
        print(classification_report(y, predict_train))
        if s.accuracy < loaded_model.score(X,y):
            res.append(1)
        else:
            res.append(0)

    res.append(accuracy)

    return json.dumps(res)



def collabwithdata(id, type):
    global ether_costs
    submission = Submission.objects.filter(contract=id)
    req = Request.objects.get(id=type)
    accuracy = 0
    res = []
    global training_data
    d = mergedatasets(req.dataset, training_data)
    for s in submission:
        loaded_model = joblib.load("finalized_model.sav")

        y = d[0].iloc[:, 2]
        X = d[0].iloc[:, :2]

        loaded_model.fit(X,y)

        y = d[0].iloc[:, 2]
        X = d[0].iloc[:, :2]
        predict_test = loaded_model.predict(X)
        accuracy += loaded_model.score(X,y)

        print(confusion_matrix(y, predict_test))
        print(classification_report(y, predict_test))
        if s.accuracy < loaded_model.score(X,y):
            res.append(1)
        else:
            res.append(0)

    res.append(accuracy)

    return json.dumps(res)


@task(name='reopen_contract')
def reopen():
    print("Reopening contract")
    print("Ether withdrawn: 5")
    return


@task(name='collab_without_data')
def collab_without_data():
    print("80% of reward amount will be withrawn at the end")
    return


@task(name='model_download')
def model_download():
    print("Downloading model. Ether Withdrawn: 5")
    return


def approve_request(request):
    req = Request.objects.get(id=int(request.POST['id']))
    # if req.request_type == 0 or req.request_type == 1:

    req.viewed = True

    if request.POST['approve'] == 'yes':
        req.approved = True

    else:
        req.approved = False
        req.save()
        return redirect('/accounts/profile/')

    if req.request_type == 2:
        contract = SmartContract.objects.get(id=req.contract.id)
        contract.contract_active = True
        contract.save()
        test_contracts.apply_async(queue='default', kwargs={'user': request.user.id, 'id': contract.id},
                                   countdown=300)

        start_new_contract.apply_async(queue='default', args=(request.user.id, contract.id))
    elif req.request_type == 0:
        collab_without_data.apply_async(queue='medium')
    elif req.request_type == 1:
        task = collab_with_data.apply_async(queue='medium', kwargs={'id': req.contract.id,'type':req.id})
    elif req.request_type == 3:
        reopen.apply_async(queue='medium')
        import webbrowser
        submission = Submission.objects.get(user=req.contract.user, contract=req.contract)
        webbrowser.open('http://localhost:8080/ipfs/'+submission.model)

    return redirect('/accounts/profile/')


@task(name="contract")
def start_new_contract(user, id):
    global ml_exchange, scd, w_scale, b_scale, web3, timeout, total_gas_used, pending_tasks, chain
    print("Web3 provider is", web3.providers)

    _hashed_data_groups = []
    accuracy_criteria = 5000  # 50.00%
    total_gas_used = 0
    timeout = 180
    w_scale = 1000  # Scale up weights by 1000x
    b_scale = 1000  # Scale up biases by 1000x

    # Load Populus contract proxy classes
    print("Web3 provider is", web3.providers)

    contract = SmartContract.objects.get(id=id)
    offer_account = web3.eth.accounts[user]
    user = User.objects.get(id=user)

    # solver_account = web3.eth.accounts[1]
    web3.eth.defaultAccount = offer_account

    scd = DemoDataset(training_percentage=0.8, partition_size=25)
    scd.generate_nonce()
    scd.sha_all_data_groups()

    print("All data groups: " + str(scd.data))
    print("All nonces: " + str(scd.nonce))

    # Initialization step 1
    print("Hashed data groups: " + str(scd.hashed_data_group))
    print("Hashed Hex data groups: " +
          str(list(map(lambda x: "0x" + x.hex(), scd.hashed_data_group))))

    print("Starting block: " + str(web3.eth.blockNumber))
    days = contract.submission_period_days * 3600 * 24
    hours = contract.submission_period_hours * 3600
    minutes = contract.submission_period_minutes * 60
    time = days + hours + minutes

    x = ml_exchange.transact({'from': offer_account}).init1(scd.hashed_data_group,
                                                            accuracy_criteria,
                                                            offer_account, time, id - 1)

    fund_receipt = wait_for_transaction_receipt(web3, x, timeout=timeout)
    total_gas_used += fund_receipt["gasUsed"]
    print("Fund gas: " + str(fund_receipt["gasUsed"]))

    defaultUsdRate = ml_exchange.call().init_level(id - 1)
    print(defaultUsdRate)

    init1_block_number = web3.eth.blockNumber
    print("Init1 block: " + str(init1_block_number))

    chain.wait.for_block(init1_block_number + 1)
    dgi = []
    init2_block_number = web3.eth.blockNumber
    print("Init2 block: " + str(init2_block_number))

    for i in range(scd.num_data_groups):
        dgi.append(i)

    print("Data group indexes: " + str(dgi))

    init2_tx = ml_exchange.transact({'from': offer_account}).init2(id - 1)
    init2_receipt = wait_for_transaction_receipt(web3, init2_tx, timeout=timeout)
    total_gas_used += init2_receipt["gasUsed"]
    print("Init2 gas: " + str(init2_receipt["gasUsed"]))
    chain.wait.for_receipt(init2_tx)

    # Can only access one element of a public array at a time
    training_partition = list(map(lambda x: ml_exchange.call().training_partition(id - 1, x), \
                                  range(scd.num_train_data_groups)))
    testing_partition = list(map(lambda x: ml_exchange.call().testing_partition(id - 1, x), \
                                 range(scd.num_test_data_groups)))
    # get partitions
    print("Training partition: " + str(training_partition))
    print("Testing partition: " + str(testing_partition))

    scd.partition_dataset(training_partition, testing_partition)
    # Initialization step 3
    # Time to reveal the training dataset
    training_nonces = []
    training_data = []
    for i in training_partition:
        training_nonces.append(scd.nonce[i])
    # Pack data into a 1-dimension array
    # Since the data array is too large, we're going to send them in single data group chunks

    train_data = scd.pack_data(scd.train_data)
    init3_tx = []
    for i in range(len(training_partition)):
        start = i * scd.dps * scd.partition_size
        end = start + scd.dps * scd.partition_size
        print(
            "(" + str(training_partition[i]) + ") Train data nonce: " + str(train_data[start:end]) + "," + str(
                scd.train_nonce[i]))
        iter_tx = ml_exchange.transact({'from': offer_account}).init3(train_data[start:end], scd.train_nonce[i],
                                                                      id - 1)
        iter_receipt = wait_for_transaction_receipt(web3, iter_tx, timeout=timeout)
        total_gas_used += iter_receipt["gasUsed"]
        print("Reveal train data iter " + str(i) + " gas: " + str(iter_receipt["gasUsed"]))
        init3_tx.append(iter_tx)
        chain.wait.for_receipt(init3_tx[i])
        print("Training Data revealed", ml_exchange.call().train_dg_revealed(id - 1))

    init3_block_number = web3.eth.blockNumber
    print("Init3 block: " + str(init3_block_number))

    contract = SmartContract.objects.get(id=id)
    contract.submission_phase = True
    contract.save()

    return


def wait_for_testing(user, id, amt):
    time.sleep(120)
    test_contracts.apply_async(queue='default', kwargs={'user': user.id, 'id': id})


def home_page(request):
    simulation.apply_async(queue='default')
    # classification = Classification([],2)
    contracts = SmartContract.objects.all()
    to_requests = Request.objects.filter(to_user=request.user)

    from_requests = Request.objects.filter(from_user=request.user)
    for c in contracts:
        submission = Submission.objects.filter(contract=c.id, user=request.user)
        if len(submission) == 0:
            c.train_data = True
        else:
            c.train_data = False
            if submission[0].model:
                c.model = True
                c.accuracy = submission[0].accuracy
            else:
                c.model = False
    return render(request, 'ml_exchange/index.html', {'request': request, 'contracts': contracts, 'to_requests':to_requests, 'from_requests':from_requests})


def new_contract(request):
    if request.method == 'POST':
        contract = SmartContract()
        contract.name = request.POST['name']
        contract.user = request.user
        contract.submission_period_days = int(request.POST['days'])
        contract.submission_period_hours = int(request.POST['hours'])
        contract.submission_period_minutes = int(request.POST['minutes'])
        contract.document = request.FILES['document']
        contract.contract_active = True
        contract.save()

        days = contract.submission_period_days * 3600 * 24
        hours = contract.submission_period_hours * 3600
        minutes = contract.submission_period_minutes * 60
        time_to_test = days + hours + minutes

        print(time_to_test)

        test_contracts.apply_async(queue='default', kwargs={'user': request.user.id, 'id': contract.id}, countdown=time_to_test)

        start_new_contract.apply_async(queue='default', args=(request.user.id, contract.id))

        return redirect('/accounts/profile/')


def login(request):
    if not request.user.is_authenticated:
        return render(request, 'web3auth/login.html')
    else:
        return redirect('/accounts/profile/')


def mergedatasets(X,Y):
    res = []
    res[0].append(Y.iloc[:200],X)
    res[1].append(Y.iloc[:200:500])
    return res


def auto_login(request):
    if not request.user.is_authenticated:
        return render(request, 'web3auth/autologin.html')
    else:
        return redirect('/accounts/profile/')


@task(name = "simulation")
def simulation():
    global training_dataset, total_gas_used
    contract = SmartContract()
    contract.name = 'Election'
    contract.user = User.objects.get(id=2)
    contract.submission_period_days = 0
    contract.submission_period_hours = 0
    contract.submission_period_minutes = 4
    contract.contract_active = True
    contract.document = 'QmTjtkvkYuWscQzqjHHFyBFir36CGbxoCBCbWuHeBHiSfG'
    contract.save()

    days = contract.submission_period_days * 3600 * 24
    hours = contract.submission_period_hours * 3600
    minutes = contract.submission_period_minutes * 60
    time_to_test = days + hours + minutes

    print(time_to_test)

    #test_contracts.apply_async(queue='default', kwargs={'user': 1, 'id': contract.id},countdown=time_to_test)

    new_contract = start_new_contract.apply_async(queue='default', args=(2, contract.id))
    while new_contract.state != "SUCCESS":
        continue

    id = contract.id
    print("User: "+str(User.objects.get(id=3))+" accessing training data")
    print("Contract training data length: "+str(ml_exchange.call().get_train_data_length(id - 1)))

    print("User: " + str(User.objects.get(id=4)) + " accessing training data")
    print("Contract training data length: "+str(ml_exchange.call().get_train_data_length(id - 1)))

    contract_train_data = []
    contract_train_data_length = ml_exchange.call().get_train_data_length(id - 1)
    for i in range(contract_train_data_length):
        for j in range(scd.dps):
            contract_train_data.append(ml_exchange.call().train_data(id - 1, i, j))
    training_dataset = contract_train_data
    contract_train_data = scd.unpack_data(contract_train_data)
    print("Contract training data: " + str(contract_train_data))


    import csv
    csvfile = open('/home/jessica/ml_exchange_project/data/test_data.csv', 'w', newline='\n')
    obj = csv.writer(csvfile)
    for data in contract_train_data:
        obj.writerow(data)
    csvfile.close()

    submission = Submission()
    submission.contract = SmartContract.objects.get(id=contract.id)
    submission.user = User.objects.get(id=3)
    submission.save()

    submission = Submission()
    submission.contract = SmartContract.objects.get(id=contract.id)
    submission.user = User.objects.get(id=4)
    submission.save()

    import webbrowser
    webbrowser.open('/home/jessica/ml_exchange_project/data/test_data.csv')

    time.sleep(2)
    import webbrowser
    webbrowser.open('/home/jessica/ml_exchange_project/data/test_data.csv')

    print("User: " + str(User.objects.get(id=3)) + " submitting model")

    print("User: " + str(User.objects.get(id=4)) + " submitting model")

    submission = Submission.objects.filter(user=3, contract=contract.id)
    s = Submission.objects.get(id=submission[0].id)
    s.model = 'QmUX5HznL9grUn4wkUQFYyAAR9uzQ7jh8D4eQx1QgQcHoh'
    s.save()
    #x = submit_model.apply_async(queue='default', kwargs={'user': 3, 'id': contract.id})
    #pending_tasks.append(x)

    submission = Submission.objects.filter(user=4, contract=contract.id)
    s = Submission.objects.get(id=submission[0].id)
    s.model = 'QmVmq91eRTMryLXmzX9yGUDWuG7dCYJRRy2jwBAzdHiWJg'
    s.save()
    #y = submit_model.apply_async(queue='default', kwargs={'user': 4, 'id': contract.id})
    #pending_tasks.append(y)


    print("Solver address: " + str(User.objects.get(id=3).username))
    task1 = run_model.apply_async(queue='low', args=(0, training_dataset))
    print("Solver address: " + str(User.objects.get(id=4).username))
    task2 = run_model.apply_async(queue='low', args=(1, training_dataset))

    while task1.state != 'SUCCESS':
        continue
    nn = task1.result
    nn = json.loads(nn)

    il_nn = 2
    hl_nn = []
    ol_nn = 1
    # Submit the solution to the contract
    submit_tx = ml_exchange.transact().submit_model(web3.eth.accounts[3], il_nn, ol_nn, hl_nn, \
                                                    nn["weights"],
                                                    nn["biases"], id - 1)
    submission_tasks.append(submit_tx)
    submit_receipt = wait_for_transaction_receipt(web3, submit_tx, timeout=timeout)
    total_gas_used += submit_receipt["gasUsed"]
    print("Submit gas: " + str(submit_receipt["gasUsed"]))
    chain.wait.for_receipt(submit_tx)

    # Get submission index ID
    submission_id = ml_exchange.call().get_submission_id(web3.eth.accounts[3], il_nn, \
                                                         ol_nn, hl_nn, nn["weights"], nn["biases"], id - 1)

    submission = Submission.objects.filter(contract=id, user=3)
    s = Submission.objects.get(id=submission[0].id)
    s.accuracy = nn["accuracy"]
    s.save()
    print("Submission ID: " + str(submission_id))

    while task2.state != 'SUCCESS':
        continue
    nn = task2.result
    nn = json.loads(nn)

    il_nn = 2
    hl_nn = [4, 4, 4]
    ol_nn = 1

    submit_tx = ml_exchange.transact().submit_model(web3.eth.accounts[4], il_nn, ol_nn, hl_nn, nn["weights"], nn["biases"], id - 1)
    submission_tasks.append(submit_tx)
    submit_receipt = wait_for_transaction_receipt(web3, submit_tx, timeout=timeout)
    total_gas_used += submit_receipt["gasUsed"]
    print("Submit gas: " + str(submit_receipt["gasUsed"]))
    chain.wait.for_receipt(submit_tx)

    # Get submission index ID
    submission_id = ml_exchange.call().get_submission_id(web3.eth.accounts[4], il_nn, ol_nn, hl_nn, nn["weights"], nn["biases"], id - 1)
    submission = Submission.objects.filter(contract=id, user=4)
    s = Submission.objects.get(id=submission[0].id)
    s.accuracy = nn["accuracy"]
    s.save()
    print("Submission ID: " + str(submission_id))

    time.sleep(20)
    print("Dataset Collaboration 1")

    reqs = []
    req1 = Request()
    req1.from_user = User.objects.get(id=5)
    req1.to_user = contract.user
    req1.contract = contract
    req1.request_type = 1
    req1.dataset = 'QmQpDnEcNLPCBUtkREC9AexgQKhY9DiV866Rz1qwJEaowS'
    req1.save()

    print("Dataset Collaboration 2")

    req2 = Request()
    req2.from_user = User.objects.get(id=3)
    req2.to_user = contract.user
    req2.contract = contract
    req2.request_type = 1
    req2.dataset = 'QmUX5HznL9grUn4wkUQFYyAAR9uzQ7jh8D4eQx1QgQcHoh'
    req2.save()

    reqs.append(req1)
    reqs.append(req2)

    pt = []
    for req in reqs:

        req.viewed = True
        req.approved = True

        if req.request_type == 2:
            contract = SmartContract.objects.get(id=req.contract.id)
            contract.contract_active = True
            contract.save()
            test_contracts.apply_async(queue='default', kwargs={'user': contract.user, 'id': contract.id},
                                       countdown=300)

            start_new_contract.apply_async(queue='default', args=(contract.user, contract.id))
        elif req.request_type == 0:
            collab_without_data.apply_async(queue='medium')
        elif req.request_type == 1:
            task = collab_with_data.apply_async(queue='medium', kwargs={'id': req.contract.id, 'type': req.id})
            pt.append(task)
        elif req.request_type == 3:
            reopen.apply_async(queue='medium')
            import webbrowser
            submission = Submission.objects.get(user=req.contract.user, contract=req.contract)
            webbrowser.open('http://localhost:8080/ipfs/' + submission.model)

        req.save()

    import random
    max = 0
    max = 1
    while pt[0].state != 'SUCCESS':
        continue
    while pt[1].state != 'SUCCESS':
        continue
    a = 0
    cnt1 =0
    cnt2= 0
    sub = Submission.objects.filter(contract=contract.id)
    for s in sub:
        a = a+s.accuracy

    res1 = json.loads(pt[0].result)
    res2 = json.loads(pt[1].result)
    print(a)
    if (res1[0] == 1 and res1[1] == 1) or ((res1[0]==1 or res1[1] == 1)):
        print("Request: 1")
        r = random.randint(601, 799) / 100
        print("Calculating ether: ", r)
    else:
        print("Request: 1")
        r = random.randint(301, 499) / 100
        print("Calculating ether: ", r)
    if (res2[0] == 1 and res2[1] == 1) or ((res2[0]==1 or res2[1] == 1) ):
        print("Request: 2")
        r = random.randint(601, 799) / 100
        print("Calculating ether: ", r)
    else:
        print("Request: 2")
        r = random.randint(301, 499) / 100
        print("Calculating ether: ", r)

    time.sleep(20)
    t = test_contracts.apply_async(queue='default', kwargs={'user': contract.user.id, 'id': contract.id})
    while t.state != 'SUCCESS':
        continue

    time.sleep(20)

    reqs = []
    req1 = Request()
    req1.from_user = User.objects.get(id=5)
    req1.to_user = contract.user
    req1.contract = contract
    req1.request_type = 3
    req1.document = 'QmUX5HznL9grUn4wkUQFYyAAR9uzQ7jh8D4eQx1QgQcHoh'
    req1.save()

    reopen.apply_async(queue='medium')
    import webbrowser
    webbrowser.open('/home/jessica/ml_exchange_project/dutils/classification_2.py')

    contract = SmartContract()
    contract.name = 'Election'
    contract.user = User.objects.get(id=2)
    contract.submission_period_days = 0
    contract.submission_period_hours = 0
    contract.submission_period_minutes = 4
    contract.contract_active = True
    contract.document = 'QmTjtkvkYuWscQzqjHHFyBFir36CGbxoCBCbWuHeBHiSfG'
    contract.save()

    days = contract.submission_period_days * 3600 * 24
    hours = contract.submission_period_hours * 3600
    minutes = contract.submission_period_minutes * 60
    time_to_test = days + hours + minutes

    print(time_to_test)
    #test_contracts.apply_async(queue='default', kwargs={'user': contract.user.id, 'id': contract.id},countdown=300)

    new_contract = start_new_contract.apply_async(queue='default', args=(contract.user.id, contract.id))
    while new_contract.state != "SUCCESS":
        continue

    id = contract.id
    print("User: "+str(User.objects.get(id=5))+" accessing training data")
    print("Contract training data length: "+str(ml_exchange.call().get_train_data_length(id - 1)))

    contract_train_data = []
    contract_train_data_length = ml_exchange.call().get_train_data_length(id - 1)
    for i in range(contract_train_data_length):
        for j in range(scd.dps):
            contract_train_data.append(ml_exchange.call().train_data(id - 1, i, j))
    training_dataset = contract_train_data
    contract_train_data = scd.unpack_data(contract_train_data)
    print("Contract training data: " + str(contract_train_data))

    submission = Submission()
    submission.contract = SmartContract.objects.get(id=contract.id)
    submission.user = User.objects.get(id=5)
    submission.save()

    print("User: " + str(User.objects.get(id=5)) + " submitting model")

    submission = Submission.objects.filter(user=5, contract=contract.id)
    s = Submission.objects.get(id=submission[0].id)
    s.model = 'QmdiyKrFS3hLteDGnGYXQD3p1qoCiJWKrtXcYNQWxAL2pG'
    s.save()

    print("Solver address: " + str(User.objects.get(id=5).username))
    task1 = run_model.apply_async(queue='low', args=(0, training_dataset))

    while task1.state != 'SUCCESS':
        continue
    nn = task1.result
    nn = json.loads(nn)

    il_nn = 2
    hl_nn = []
    ol_nn = 1
    # Submit the solution to the contract
    submit_tx = ml_exchange.transact().submit_model(web3.eth.accounts[5], il_nn, ol_nn, hl_nn, nn["weights"],
                                                    nn["biases"], id - 1)
    submission_tasks.append(submit_tx)
    submit_receipt = wait_for_transaction_receipt(web3, submit_tx, timeout=timeout)
    total_gas_used += submit_receipt["gasUsed"]
    print("Submit gas: " + str(submit_receipt["gasUsed"]))
    chain.wait.for_receipt(submit_tx)

    # Get submission index ID
    submission_id = ml_exchange.call().get_submission_id(web3.eth.accounts[5], il_nn,ol_nn, hl_nn, nn["weights"], nn["biases"], id - 1)

    submission = Submission.objects.filter(contract=id, user=5)
    s = Submission.objects.get(id=submission[0].id)
    s.accuracy = nn["accuracy"]
    s.save()
    #print("Submission ID: " + str(submission_id))
    print("Collaboration contribution:"+str(nn["accuracy"]*100/3))
    return


def _simulation():
    global training_dataset, total_gas_used
    contract = SmartContract()
    contract.name = 'Election'
    contract.user = User.objects.get(id=2)
    contract.submission_period_days = 0
    contract.submission_period_hours = 0
    contract.submission_period_minutes = 4
    contract.contract_active = True
    contract.document = 'QmTjtkvkYuWscQzqjHHFyBFir36CGbxoCBCbWuHeBHiSfG'
    contract.save()

    days = contract.submission_period_days * 3600 * 24
    hours = contract.submission_period_hours * 3600
    minutes = contract.submission_period_minutes * 60
    time_to_test = days + hours + minutes

    print(time_to_test)

    #test_contracts.apply_async(queue='default', kwargs={'user': 1, 'id': contract.id},countdown=time_to_test)

    new_contract = start_new_contract.apply_async(queue='default', args=(2, contract.id))
    while new_contract.state != "SUCCESS":
        continue

    id = contract.id
    print("User: "+str(User.objects.get(id=3))+" accessing training data")
    print("Contract training data length: "+str(ml_exchange.call().get_train_data_length(id - 1)))

    print("User: " + str(User.objects.get(id=4)) + " accessing training data")
    print("Contract training data length: "+str(ml_exchange.call().get_train_data_length(id - 1)))

    contract_train_data = []
    contract_train_data_length = ml_exchange.call().get_train_data_length(id - 1)
    for i in range(contract_train_data_length):
        for j in range(scd.dps):
            contract_train_data.append(ml_exchange.call().train_data(id - 1, i, j))
    training_dataset = contract_train_data
    contract_train_data = scd.unpack_data(contract_train_data)
    print("Contract training data: " + str(contract_train_data))


    import csv
    csvfile = open('/home/jessica/ml_exchange_project/data/test_data.csv', 'w', newline='\n')
    obj = csv.writer(csvfile)
    for data in contract_train_data:
        obj.writerow(data)
    csvfile.close()

    submission = Submission()
    submission.contract = SmartContract.objects.get(id=contract.id)
    submission.user = User.objects.get(id=3)
    submission.save()

    submission = Submission()
    submission.contract = SmartContract.objects.get(id=contract.id)
    submission.user = User.objects.get(id=4)
    submission.save()

    import webbrowser
    webbrowser.open('/home/jessica/ml_exchange_project/data/test_data.csv')

    time.sleep(2)
    import webbrowser
    webbrowser.open('/home/jessica/ml_exchange_project/data/test_data.csv')

    print("User: " + str(User.objects.get(id=3)) + " submitting model")

    print("User: " + str(User.objects.get(id=4)) + " submitting model")

    submission = Submission.objects.filter(user=3, contract=contract.id)
    s = Submission.objects.get(id=submission[0].id)
    s.model = 'QmUX5HznL9grUn4wkUQFYyAAR9uzQ7jh8D4eQx1QgQcHoh'
    s.save()
    #x = submit_model.apply_async(queue='default', kwargs={'user': 3, 'id': contract.id})
    #pending_tasks.append(x)

    submission = Submission.objects.filter(user=4, contract=contract.id)
    s = Submission.objects.get(id=submission[0].id)
    s.model = 'QmVmq91eRTMryLXmzX9yGUDWuG7dCYJRRy2jwBAzdHiWJg'
    s.save()
    #y = submit_model.apply_async(queue='default', kwargs={'user': 4, 'id': contract.id})
    #pending_tasks.append(y)


    print("Solver address: " + str(User.objects.get(id=3).username))
    task1 = run_model.apply_async(queue='low', args=(0, training_dataset))
    print("Solver address: " + str(User.objects.get(id=4).username))
    task2 = run_model.apply_async(queue='low', args=(1, training_dataset))

    while task1.state != 'SUCCESS':
        continue
    nn = task1.result
    nn = json.loads(nn)

    il_nn = 2
    hl_nn = []
    ol_nn = 1
    # Submit the solution to the contract
    submit_tx = ml_exchange.transact().submit_model(web3.eth.accounts[3], il_nn, ol_nn, hl_nn, \
                                                    nn["weights"],
                                                    nn["biases"], id - 1)
    submission_tasks.append(submit_tx)
    submit_receipt = wait_for_transaction_receipt(web3, submit_tx, timeout=timeout)
    total_gas_used += submit_receipt["gasUsed"]
    print("Submit gas: " + str(submit_receipt["gasUsed"]))
    chain.wait.for_receipt(submit_tx)

    # Get submission index ID
    submission_id = ml_exchange.call().get_submission_id(web3.eth.accounts[3], il_nn, \
                                                         ol_nn, hl_nn, nn["weights"], nn["biases"], id - 1)

    submission = Submission.objects.filter(contract=id, user=3)
    s = Submission.objects.get(id=submission[0].id)
    s.accuracy = nn["accuracy"]
    s.save()
    print("Submission ID: " + str(submission_id))

    while task2.state != 'SUCCESS':
        continue
    nn = task2.result
    nn = json.loads(nn)

    il_nn = 2
    hl_nn = [4, 4, 4]
    ol_nn = 1

    submit_tx = ml_exchange.transact().submit_model(web3.eth.accounts[4], il_nn, ol_nn, hl_nn, nn["weights"], nn["biases"], id - 1)
    submission_tasks.append(submit_tx)
    submit_receipt = wait_for_transaction_receipt(web3, submit_tx, timeout=timeout)
    total_gas_used += submit_receipt["gasUsed"]
    print("Submit gas: " + str(submit_receipt["gasUsed"]))
    chain.wait.for_receipt(submit_tx)

    # Get submission index ID
    submission_id = ml_exchange.call().get_submission_id(web3.eth.accounts[4], il_nn, ol_nn, hl_nn, nn["weights"], nn["biases"], id - 1)
    submission = Submission.objects.filter(contract=id, user=4)
    s = Submission.objects.get(id=submission[0].id)
    s.accuracy = nn["accuracy"]
    s.save()
    print("Submission ID: " + str(submission_id))

    time.sleep(20)
    print("Dataset Collaboration 1")

    reqs = []
    req1 = Request()
    req1.from_user = User.objects.get(id=5)
    req1.to_user = contract.user
    req1.contract = contract
    req1.request_type = 1
    req1.dataset = 'QmQpDnEcNLPCBUtkREC9AexgQKhY9DiV866Rz1qwJEaowS'
    req1.save()

    print("Dataset Collaboration 2")

    req2 = Request()
    req2.from_user = User.objects.get(id=3)
    req2.to_user = contract.user
    req2.contract = contract
    req2.request_type = 1
    req2.dataset = 'QmUX5HznL9grUn4wkUQFYyAAR9uzQ7jh8D4eQx1QgQcHoh'
    req2.save()

    reqs.append(req1)
    reqs.append(req2)

    pt = []
    for req in reqs:

        req.viewed = True
        req.approved = True

        if req.request_type == 2:
            contract = SmartContract.objects.get(id=req.contract.id)
            contract.contract_active = True
            contract.save()
            test_contracts.apply_async(queue='default', kwargs={'user': contract.user, 'id': contract.id},
                                       countdown=300)

            start_new_contract.apply_async(queue='default', args=(contract.user, contract.id))
        elif req.request_type == 0:
            collab_without_data.apply_async(queue='medium')
        elif req.request_type == 1:
            task = collab_with_data.apply_async(queue='medium', kwargs={'id': req.contract.id, 'type': req.id})
            pt.append(task)
        elif req.request_type == 3:
            reopen.apply_async(queue='medium')
            import webbrowser
            submission = Submission.objects.get(user=req.contract.user, contract=req.contract)
            webbrowser.open('http://localhost:8080/ipfs/' + submission.model)

        req.save()

    import random
    max = 0
    max = 1
    while pt[0].state != 'SUCCESS':
        continue
    while pt[1].state != 'SUCCESS':
        continue
    a = 0
    cnt1 =0
    cnt2= 0
    sub = Submission.objects.filter(contract=contract.id)
    for s in sub:
        a = a+s.accuracy

    res1 = json.loads(pt[0].result)
    res2 = json.loads(pt[1].result)

    time.sleep(20)
    t = test_contracts.apply_async(queue='default', kwargs={'user': contract.user.id, 'id': contract.id})
    while t.state != 'SUCCESS':
        continue

    time.sleep(20)

    reqs = []
    req1 = Request()
    req1.from_user = User.objects.get(id=5)
    req1.to_user = contract.user
    req1.contract = contract
    req1.request_type = 3
    req1.document = 'QmUX5HznL9grUn4wkUQFYyAAR9uzQ7jh8D4eQx1QgQcHoh'
    req1.save()

    reopen.apply_async(queue='medium')
    import webbrowser
    webbrowser.open('/home/jessica/ml_exchange_project/dutils/classification_2.py')

    contract = SmartContract()
    contract.name = 'Election'
    contract.user = User.objects.get(id=2)
    contract.submission_period_days = 0
    contract.submission_period_hours = 0
    contract.submission_period_minutes = 4
    contract.contract_active = True
    contract.document = 'QmTjtkvkYuWscQzqjHHFyBFir36CGbxoCBCbWuHeBHiSfG'
    contract.save()

    days = contract.submission_period_days * 3600 * 24
    hours = contract.submission_period_hours * 3600
    minutes = contract.submission_period_minutes * 60
    time_to_test = days + hours + minutes

    print(time_to_test)
    #test_contracts.apply_async(queue='default', kwargs={'user': contract.user.id, 'id': contract.id},countdown=300)

    new_contract = start_new_contract.apply_async(queue='default', args=(contract.user.id, contract.id))
    while new_contract.state != "SUCCESS":
        continue

    id = contract.id
    print("User: "+str(User.objects.get(id=5))+" accessing training data")
    print("Contract training data length: "+str(ml_exchange.call().get_train_data_length(id - 1)))

    contract_train_data = []
    contract_train_data_length = ml_exchange.call().get_train_data_length(id - 1)
    for i in range(contract_train_data_length):
        for j in range(scd.dps):
            contract_train_data.append(ml_exchange.call().train_data(id - 1, i, j))
    training_dataset = contract_train_data
    contract_train_data = scd.unpack_data(contract_train_data)
    print("Contract training data: " + str(contract_train_data))

    submission = Submission()
    submission.contract = SmartContract.objects.get(id=contract.id)
    submission.user = User.objects.get(id=5)
    submission.save()

    print("User: " + str(User.objects.get(id=5)) + " submitting model")

    submission = Submission.objects.filter(user=5, contract=contract.id)
    s = Submission.objects.get(id=submission[0].id)
    s.model = 'QmdiyKrFS3hLteDGnGYXQD3p1qoCiJWKrtXcYNQWxAL2pG'
    s.save()

    print("Solver address: " + str(User.objects.get(id=5).username))
    task1 = run_model.apply_async(queue='low', args=(0, training_dataset))

    while task1.state != 'SUCCESS':
        continue
    nn = task1.result
    nn = json.loads(nn)

    il_nn = 2
    hl_nn = []
    ol_nn = 1
    # Submit the solution to the contract
    submit_tx = ml_exchange.transact().submit_model(web3.eth.accounts[5], il_nn, ol_nn, hl_nn, nn["weights"],
                                                    nn["biases"], id - 1)
    submission_tasks.append(submit_tx)
    submit_receipt = wait_for_transaction_receipt(web3, submit_tx, timeout=timeout)
    total_gas_used += submit_receipt["gasUsed"]
    print("Submit gas: " + str(submit_receipt["gasUsed"]))
    chain.wait.for_receipt(submit_tx)

    # Get submission index ID
    submission_id = ml_exchange.call().get_submission_id(web3.eth.accounts[5], il_nn,ol_nn, hl_nn, nn["weights"], nn["biases"], id - 1)

    submission = Submission.objects.filter(contract=id, user=5)
    s = Submission.objects.get(id=submission[0].id)
    s.accuracy = nn["accuracy"]
    s.save()
    #print("Submission ID: " + str(submission_id))
    return


def logout_request(request):
    logout(request)
    return redirect('/login/')