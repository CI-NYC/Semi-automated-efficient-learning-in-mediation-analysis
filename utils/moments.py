import torch
import numpy as np

# Returns the moment for the ATE example, for each sample in x
def ate_moment_fn(x, test_fn, device):
    if torch.is_tensor(x):
        with torch.no_grad():
            t1 = torch.cat([torch.ones((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
            t0 = torch.cat([torch.zeros((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
    else:
        t1 = np.hstack([np.ones((x.shape[0], 1)), x[:, 1:]])
        t0 = np.hstack([np.zeros((x.shape[0], 1)), x[:, 1:]])
    return test_fn(t1) - test_fn(t0)

# Returns the moment for the NDE example, this is the part of theta2
# theta2 is a function of (a, m, w)
# We assume the data structure is (A, M, W), with A being the treatment
# M being the mediator, and W being the covariates.

def nde_theta2_moment_fn(x, test_fn, device, a_1 = 1):
    if torch.is_tensor(x):
        with torch.no_grad():
            if a_1 == 1:
                t1 = torch.cat([torch.ones((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
            elif a_1 == 0:
                t1 = torch.cat([torch.zeros((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
    else:
        if a_1 == 1:
            t1 = np.hstack([np.ones((x.shape[0], 1)), x[:, 1:]])
        elif a_1 == 0:
            t0 = np.hstack([np.zeros((x.shape[0], 1)), x[:, 1:]])
    return test_fn(t1)

# Returns the moment for the NDE example, this is the part of theta1
# theta1 is a function of (a, w)
# We assume the data structure is (A, M, W), with A being the treatment
# M being the mediator, and W being the covariates.

def nde_theta1_moment_fn(x, test_fn, device, a_2 = 0):
    if torch.is_tensor(x):
        with torch.no_grad():
            if a_2 == 1:
                t1 = torch.cat([torch.ones((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
            elif a_2 == 0:
                t1 = torch.cat([torch.zeros((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
    else:
        if a_2 == 1:
            t1 = np.hstack([np.ones((x.shape[0], 1)), x[:, 1:]])
        elif a_2 == 0:
            t0 = np.hstack([np.zeros((x.shape[0], 1)), x[:, 1:]])
    return test_fn(t1)


# Returns the moment for the NDE example, this is the part of theta2
# theta3 is a function of (a, z, m, w)
# We assume the data structure is (A, Z, M, W), with A being the treatment
# M being the mediator, Z being the intermediate confounder, and W being the covariates.

def rde_rie_theta3_moment_fn(x, test_fn, device, a_prime = 1):
    if torch.is_tensor(x):
        with torch.no_grad():
            if a_prime == 1:
                t1 = torch.cat([torch.ones((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
            elif a_prime == 0:
                t1 = torch.cat([torch.zeros((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
    else:
        if a_prime == 1:
            t1 = np.hstack([np.ones((x.shape[0], 1)), x[:, 1:]])
        elif a_prime == 0:
            t1 = np.hstack([np.zeros((x.shape[0], 1)), x[:, 1:]])
    return test_fn(t1)

# Returns the moment for the NDE example, this is the part of theta1
# theta1 is a function of (a, w)
# We assume the data structure is (A, M, W), with A being the treatment
# M being the mediator, and W being the covariates.

def rde_rie_theta2_M_moment_fn(x, test_fn, device, a_star = 0):
    if torch.is_tensor(x):
        with torch.no_grad():
            if a_star == 1:
                t1 = torch.cat([torch.ones((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
            elif a_star == 0:
                t1 = torch.cat([torch.zeros((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
    else:
        if a_star == 1:
            t1 = np.hstack([np.ones((x.shape[0], 1)), x[:, 1:]])
        elif a_star == 0:
            t1 = np.hstack([np.zeros((x.shape[0], 1)), x[:, 1:]])
    return test_fn(t1)

def rde_rie_theta2_Z_moment_fn(x, test_fn, device, a_prime = 0):
    if torch.is_tensor(x):
        with torch.no_grad():
            if a_prime == 1:
                t1 = torch.cat([torch.ones((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
            elif a_prime == 0:
                t1 = torch.cat([torch.zeros((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
    else:
        if a_prime == 1:
            t1 = np.hstack([np.ones((x.shape[0], 1)), x[:, 1:]])
        elif a_prime == 0:
            t1 = np.hstack([np.zeros((x.shape[0], 1)), x[:, 1:]])
    return test_fn(t1)

def rde_rie_theta1_M_moment_fn(x, test_fn, device, a_prime = 1):
    if torch.is_tensor(x):
        with torch.no_grad():
            if a_prime == 1:
                t1 = torch.cat([torch.ones((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
            elif a_prime == 0:
                t1 = torch.cat([torch.zeros((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
    else:
        if a_prime == 1:
            t1 = np.hstack([np.ones((x.shape[0], 1)), x[:, 1:]])
        elif a_prime == 0:
            t1 = np.hstack([np.zeros((x.shape[0], 1)), x[:, 1:]])
    return test_fn(t1)

def rde_rie_theta1_Z_moment_fn(x, test_fn, device, a_star = 0):
    if torch.is_tensor(x):
        with torch.no_grad():
            if a_star == 1:
                t1 = torch.cat([torch.ones((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
            elif a_star == 0:
                t1 = torch.cat([torch.zeros((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
    else:
        if a_star == 1:
            t1 = np.hstack([np.ones((x.shape[0], 1)), x[:, 1:]])
        elif a_star == 0:
            t1 = np.hstack([np.zeros((x.shape[0], 1)), x[:, 1:]])
    return test_fn(t1)

def recant_nonprime_moment_fn_1(x, test_fn, device, a_3 = 1):
    if torch.is_tensor(x):
        with torch.no_grad():
            if a_3 == 1:
                t1 = torch.cat([torch.ones((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
            elif a_3 == 0:
                t1 = torch.cat([torch.zeros((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
    else:
        if a_3 == 1:
            t1 = np.hstack([np.ones((x.shape[0], 1)), x[:, 1:]])
        elif a_3 == 0:
            t1 = np.hstack([np.zeros((x.shape[0], 1)), x[:, 1:]])
    return test_fn(t1)

def recant_nonprime_moment_fn_2(x, test_fn, device, a_2 = 0):
    if torch.is_tensor(x):
        with torch.no_grad():
            if a_2 == 1:
                t1 = torch.cat([torch.ones((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
            elif a_2 == 0:
                t1 = torch.cat([torch.zeros((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
    else:
        if a_2 == 1:
            t1 = np.hstack([np.ones((x.shape[0], 1)), x[:, 1:]])
        elif a_2 == 0:
            t1 = np.hstack([np.zeros((x.shape[0], 1)), x[:, 1:]])
    return test_fn(t1)

def recant_nonprime_moment_fn_3(x, test_fn, device, a_1 = 0):
    if torch.is_tensor(x):
        with torch.no_grad():
            if a_1 == 1:
                t1 = torch.cat([torch.ones((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
            elif a_1 == 0:
                t1 = torch.cat([torch.zeros((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
    else:
        if a_1 == 1:
            t1 = np.hstack([np.ones((x.shape[0], 1)), x[:, 1:]])
        elif a_1 == 0:
            t1 = np.hstack([np.zeros((x.shape[0], 1)), x[:, 1:]])
    return test_fn(t1)

def recant_prime_moment_fn_1_Z(x, test_fn, device, a_4 = 0):
    if torch.is_tensor(x):
        with torch.no_grad():
            if a_4 == 1:
                t1 = torch.cat([torch.ones((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
            elif a_4 == 0:
                t1 = torch.cat([torch.zeros((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
    else:
        if a_4 == 1:
            t1 = np.hstack([np.ones((x.shape[0], 1)), x[:, 1:]])
        elif a_4 == 0:
            t1 = np.hstack([np.zeros((x.shape[0], 1)), x[:, 1:]])
    return test_fn(t1)

def recant_prime_moment_fn_1_M(x, test_fn, device, a_2 = 0):
    if torch.is_tensor(x):
        with torch.no_grad():
            if a_2 == 1:
                t1 = torch.cat([torch.ones((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
            elif a_2 == 0:
                t1 = torch.cat([torch.zeros((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
    else:
        if a_2 == 1:
            t1 = np.hstack([np.ones((x.shape[0], 1)), x[:, 1:]])
        elif a_2 == 0:
            t1 = np.hstack([np.zeros((x.shape[0], 1)), x[:, 1:]])
    return test_fn(t1)

def recant_prime_moment_fn_2_Z(x, test_fn, device, a_3 = 0):
    if torch.is_tensor(x):
        with torch.no_grad():
            if a_3 == 1:
                t1 = torch.cat([torch.ones((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
            elif a_3 == 0:
                t1 = torch.cat([torch.zeros((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
    else:
        if a_3 == 1:
            t1 = np.hstack([np.ones((x.shape[0], 1)), x[:, 1:]])
        elif a_3 == 0:
            t1 = np.hstack([np.zeros((x.shape[0], 1)), x[:, 1:]])
    return test_fn(t1)

def recant_prime_moment_fn_2_M(x, test_fn, device, a_4 = 0):
    if torch.is_tensor(x):
        with torch.no_grad():
            if a_4 == 1:
                t1 = torch.cat([torch.ones((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
            elif a_4 == 0:
                t1 = torch.cat([torch.zeros((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
    else:
        if a_4 == 1:
            t1 = np.hstack([np.ones((x.shape[0], 1)), x[:, 1:]])
        elif a_4 == 0:
            t1 = np.hstack([np.zeros((x.shape[0], 1)), x[:, 1:]])
    return test_fn(t1)

def recant_prime_moment_fn_3_Z(x, test_fn, device, a_2 = 0):
    if torch.is_tensor(x):
        with torch.no_grad():
            if a_2 == 1:
                t1 = torch.cat([torch.ones((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
            elif a_2 == 0:
                t1 = torch.cat([torch.zeros((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
    else:
        if a_2 == 1:
            t1 = np.hstack([np.ones((x.shape[0], 1)), x[:, 1:]])
        elif a_2 == 0:
            t1 = np.hstack([np.zeros((x.shape[0], 1)), x[:, 1:]])
    return test_fn(t1)

def recant_prime_moment_fn_4(x, test_fn, device, a_1 = 0):
    if torch.is_tensor(x):
        with torch.no_grad():
            if a_1 == 1:
                t1 = torch.cat([torch.ones((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
            elif a_1 == 0:
                t1 = torch.cat([torch.zeros((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
    else:
        if a_1 == 1:
            t1 = np.hstack([np.ones((x.shape[0], 1)), x[:, 1:]])
        elif a_1 == 0:
            t1 = np.hstack([np.zeros((x.shape[0], 1)), x[:, 1:]])
    return test_fn(t1)


def policy_moment_gen(policy):
    def policy_moment_fn(x, test_fn, device):
        with torch.no_grad():
            if torch.is_tensor(x):
                t1 = torch.cat([torch.ones((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
                t0 = torch.cat([torch.zeros((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
            else:
                t1 = np.hstack([np.ones((x.shape[0], 1)), x[:, 1:]])
                t0 = np.hstack([np.zeros((x.shape[0], 1)), x[:, 1:]])
            p1 = policy(x)
        out1 = test_fn(t1)
        out0 = test_fn(t0)
        if len(out1.shape) > 1:
            p1 = p1.reshape(-1, 1)
        return out1 * p1 + out0 * (1 - p1)

    return policy_moment_fn


def trans_moment_gen(trans):
    def trans_moment_fn(x, test_fn, device):
        with torch.no_grad():
            if torch.is_tensor(x):
                tx = torch.cat([x[:, [0]], trans(x[:, [1]]), x[:, 2:]], dim=1)
            else:
                tx =  np.hstack([x[:, [0]], trans(x[:, [1]]), x[:, 2:]])
        return test_fn(tx) - test_fn(x)

    return trans_moment_fn


def avg_der_moment_fn(x, test_fn, device):
    if torch.is_tensor(x):
        T = torch.autograd.Variable(x[:, [0]], requires_grad=True)
        input = torch.cat([T, x[:, 1:]], dim=1)
        output = test_fn(input)
        gradients = torch.autograd.grad(outputs=output, inputs=T,
                              grad_outputs=torch.ones(output.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    else:
        raise AttributeError('Not implemented')
    return gradients

def avg_small_diff(x, test_fn, device):
    epsilon = 0.01

    if torch.is_tensor(x):
        with torch.no_grad():
            t1 = torch.cat([(x[:, [0]] + epsilon).to(device), x[:, 1:]], dim=1)
            t0 = torch.cat([(x[:, [0]] - epsilon).to(device), x[:, 1:]], dim=1)
    else:
        t1 = np.hstack([x[:, [0]] + epsilon, x[:, 1:]])
        t0 = np.hstack([x[:, [0]] - epsilon, x[:, 1:]])
    return (test_fn(t1) - test_fn(t0)) / (2*epsilon)
