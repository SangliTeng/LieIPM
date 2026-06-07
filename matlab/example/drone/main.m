clear all;clc;close all;
%% set the path
restoredefaultpath
addpath(genpath("../../casadi-utils/"))
addpath(genpath("../../utils/"))

addpath(genpath("./"))
addpath(genpath("../config"))

%% generate the files
% gen_dynamics_casadi_batch_hiperlab

%%
test_main_hiperlab
