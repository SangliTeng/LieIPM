clear;clc;close all;
%% set the path
restoredefaultpath
addpath(genpath("../../casadi-utils/"))
addpath(genpath("../../utils/"))

%% SO(3) x R3
addpath(genpath(".\SO3"))
test_main

%% SE(3)
rmpath('.\SO3')
addpath('.\SE3')
test_main