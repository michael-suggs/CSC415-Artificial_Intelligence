%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% artmap_shell_vote.m
%
% Description: a sample program of how to use voting with ARTMAP networks
%
% Authors: Suhas Chelian, Norbert Kopco
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;

numVoters = 3; % Number of Voters
voteWTA = 0; % Whether WTA compression is done for each network
             % before voting or not

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data
TRAIN_N = 8;	%// Number of training points
input = [.8,.5; .5,.2; .8,.8; .7,.1; 1,1; 1,1; .6,.4; .2,.3];
output = [2; 2; 1; 1; 1; 1; 2; 2];

TEST_N = 8;	%// Number of testing points
te_input = [.2,.9; .9,.6; .6,.6; .9,.8; .7,.5; .2,.7; .4,.9; .9,.7];
te_output = [2; 2; 1; 2; 1; 2; 2; 2];

% Uncomment these lines to use the large training sets
%load input.dat;
%load output.dat;
train = [input, output];
trainN = size(input,1);

% Uncomment these lines to use the large testing sets
%load te_input.dat;
%load te_output.dat;
test = [te_input, te_output];
testN = size(te_input,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Init
disp( 'Initializing' );
MAPTYPE = 3;
M = 2;
L = 2;
EPOCHS = 1;
MAX_F2_SIZE = 100;
defaultParams = 1;

artmap_nets = cell(1, numVoters);

for i = 1:numVoters
  artmap_nets{1}{i} = artmap_init( MAPTYPE, M, L, MAX_F2_SIZE,...
				defaultParams );
  if ( artmap_nets{1}{i}.fail == 1 )
    disp( 'artmap_init failed!' );
    quit;
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train
disp( 'Training' );
forceInputHC = 0;
forceOutputHC = 0;
verbose = 0;

train_order = cell(1,numVoters);
for i = 1:numVoters
  train_order{1}{i} = randperm( TRAIN_N ); % each network is given
                                           % a different training order
  artmap_nets{1}{i} = artmap_train_large( artmap_nets{1}{i},... 
					  train( train_order{1}{i}, :), TRAIN_N,...
					  forceInputHC, forceOutputHC, verbose, 1 ...
					  );
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test
disp( 'Testing' );
forceInputHC = 0;
forceOutputHC = 0;
verbose = 0;

Sigma = zeros(TEST_N,L,numVoters);
for i = 1:numVoters
  [artmap_nets{1}{i}, pcTmp, corTmp, SigmaTmp] = artmap_test_large( artmap_nets{1}{i},...
					     test, TEST_N,...
					     forceInputHC, forceOutputHC, verbose ...
					     );
  
   if ( voteWTA == 1 )
     [aux,idx] = max(SigmaTmp');
     SigmaTmp = full(ind2vec(idx))';
   else
     SigmaTmp = SigmaTmp;
   end
 
   Sigma(:,:,i) = SigmaTmp;
end

SigmaAdd = sum(Sigma,3); % sum across all voters
[aux,idx] = max(SigmaAdd');
idxCorr = find( test(:,M+1) == idx' ); % see where test (K) matches idx
                                       % (Kp)
corr = size(idxCorr,1);

pc = corr/testN;
disp( sprintf( 'Percent Correct: %3.2f%%', pc*100 ) );

conf = aux/numVoters; % confidence measures
disp( sprintf( 'Average Confidence: %3.2f%%', mean(conf)*100 ) );