function [artmap_net, pc, correct, bigSigma, bigY] = artmap_test_large( artmap_net, testArg, testNArg,...
						   forceInputHC, ...
						   forceOutputHC, verbose )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% artmap_test_large.m
%
% Description:  testing an ARTMAP network on a set
%
% Authors: Suhas Chelian, Norbert Kopco
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs and Default Values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% artmap_net = (nothing)
%   The artmap_net to test.
% testArg = (nothing)
%   The testing points.  The first 1:M columns are the inputs, and
%   the M+1st column is the output.
% testNArg = (nothing)
%   The number of testing points.
% forceInputHC = (nothing)
%   Whether to force input hypercubing or not.
% forceOutputHC = (nothing)
%   Whether to force proper indexing of output classes or not.
% verbose = (nothing)
%   A value of 1 will show the beginning and ending of training; a
%   value of 2 will show progress incrementally; a value of 3 will
%   show the index of each incorrectly predicted output.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Outputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% artmap_net
%   The artmap_net with the testing set attached to it
% pc
%   Percent correct.
% correct
%   Number correct.
% bigSigma
%   Confidence profile for all testing points.
% bigY
%   F2 activity for all testing points.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Unpack
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MAPTYPE = artmap_net.type;
M = artmap_net.M;
L = artmap_net.L;
EPOCHS = artmap_net.EPOCHS;
MAX_F2_SIZE = artmap_net.MAX_F2_SIZE;

TRAIN_N = artmap_net.TRAIN_N;
input = artmap_net.input;
output = artmap_net.output;
TEST_N = artmap_net.TEST_N;
te_input = artmap_net.te_input;
te_output = artmap_net.te_output;

alpha = artmap_net.alpha;
p = artmap_net.p;
beta = artmap_net.beta;
epsilon = artmap_net.epsilon;
rho_a_bar = artmap_net.rho_a_bar;
rho_ab = artmap_net.rho_ab;
Tu = artmap_net.Tu;

F0_SIZE = artmap_net.F0_SIZE;
F2_SIZE = artmap_net.F2_SIZE;

DO_WEBER = artmap_net.DO_WEBER;
DO_CBD = artmap_net.DO_CBD;
DO_TRAIN_WTA = artmap_net.DO_TRAIN_WTA;
DO_TRAIN_ICG = artmap_net.DO_TRAIN_ICG;
DO_TRAIN_IC = artmap_net.DO_TRAIN_IC;
DO_TEST_IC = artmap_net.DO_TEST_IC;
DO_TEST_WTA = artmap_net.DO_TEST_WTA;
DO_TEST_SCG = artmap_net.DO_TEST_SCG;
DO_TEST_ICG = artmap_net.DO_TEST_ICG;

DO_OLD_Tj = artmap_net.DO_OLD_Tj;
DO_KAPPA_VEC = artmap_net.DO_KAPPA_VEC;

tau_ij = artmap_net.tau_ij;
tau_ji = artmap_net.tau_ji;
c = artmap_net.c;
C = artmap_net.C;
kappa = artmap_net.kappa;
Delta = artmap_net.Delta;
rho = artmap_net.rho;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TEST_N = testNArg;
te_input = testArg(:,1:M);
te_output = testArg(:,M+1);

%// Force unit hypercubing of input
if ( forceInputHC == 1 )
 minM2 = min(te_input(:,1:M));  
 rangeM2 = max(te_input(:,1:M)) - minM2;

 if ( TEST_N ~= 1 )
   for i = 1:M
     if ( rangeM2(i) == 0 )
       if ( max(te_input(:,i)) == 0 )
	 ; % do nothing
       else
	 te_input(:,i) = (te_input(:,i)-minM2(i))/max(te_input(:, ...
							       i));
       end
     else
       te_input(:,i) = (te_input(:,i)-minM2(i))/rangeM2(i);
     end
   end
 else
   if (rangeM2 == 0)
     if ( max(te_input) == 0 )
       ; % do nothing
     else
       te_input = (te_input-minM2)/max(te_input);
     end
   else
     te_input = (te_input-minM2)/rangeM2;
   end
 end
 
end

%// Check unit hypercubing
minM = min(te_input(:,1:M));
maxM = max(te_input(:,1:M));
findminM = find( minM < 0 );
findmaxM = find( maxM > 1 );
if ( ~isempty(findminM) | ~isempty(findmaxM) )
  disp( 'Testing input is not in unit hypercube!' );
%  quit;
  fail = 1;
end

%// Force proper indexing of output classes
if ( forceOutputHC == 1 )
 findminL = find( te_output < 1 );
 findmaxL = find( te_output > L );
 if ( ~isempty(findminL) )
   te_output( findminL ) = te_output( findminL ) + (1-te_output(findminL));
 end
 if ( ~isempty(findmaxL) )
   te_output( findmaxL ) = te_output( findmaxL ) - (te_output(findmaxL)-L);
 end
end

%// Check output classes
minL = min(te_output);
maxL = max(te_output);
if ( minL < 1 | maxL > L )
  disp( 'Testing output is not indexed properlly!' );
%  quit;
  fail = 1;
end

correct = 0;
bigSigma = zeros( TEST_N, L );
bigY = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ( verbose >= 1 )
  disp( 'Testing' );
end

mark = 0; % keeps track of percent done

%Step 1 - Next iteration
for n=1:TEST_N

  %// Copy next input pattern and corresponding output category
  A = [te_input(n,:) 1-te_input(n,:)]';
  K = te_output(n);
  [Y] = artmap_test_small( artmap_net, A );
  bigY(n,:) = Y;
  
  % Step 3 - Prediction
  sigma_k = zeros(L,1);
  for j=1:C
    if ( DO_KAPPA_VEC == 0 )
      sigma_k(kappa(j)) = sigma_k(kappa(j)) + Y(j);
    else
      sigma_k(:) = sigma_k(:) + (kappa(j,:)*Y(j))';
    end
  end

  bigSigma(n,:) = sigma_k';

  K_p = find( sigma_k == max(sigma_k));
  if (size(K_p,2) > 1)
    K_p = K_p(1);
  end
        
  % Step 4 - Prediction
  if K_p == K
    correct = correct+1;
  else
    %//Testing_output
    if ( verbose >= 3 )
      disp( sprintf( 'Test pat #%4d: INcorrect prediction', n ) );
    end
  end
  
  if ( verbose == 2 & (n/TEST_N) > (mark+.1) )
    mark = n/TEST_N;
    disp( sprintf( '  Percent done: %5.2f%%', mark*100 ) );
  end
  
end  

pc = correct/TEST_N;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pack
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DO NOT PACK HERE BECAUSE THINGS ARE LOCALLY UNCHANGED!, except te_input

%artmap_net.type = MAPTYPE;
%artmap_net.M = M;
%artmap_net.L = L;
%artmap_net.EPOCHS = EPOCHS;
%artmap_net.MAX_F2_SIZE = MAX_F2_SIZE;

%artmap_net.TRAIN_N = TRAIN_N;
%artmap_net.input = input;
%artmap_net.output = output;
artmap_net.TEST_N = TEST_N;
artmap_net.te_input = te_input;
artmap_net.te_output = te_output;

%artmap_net.alpha = alpha;
%artmap_net.p = p;
%artmap_net.beta = beta;
%artmap_net.epsilon = epsilon;
%artmap_net.rho_a_bar = rho_a_bar;
%artmap_net.Tu = Tu;

%artmap_net.F0_SIZE = M*2;
%artmap_net.F2_SIZE = MAX_F2_SIZE;

% artmap_net.DO_WEBER = DO_WEBER;
% artmap_net.DO_CBD = DO_CBD;
% artmap_net.DO_TRAIN_WTA = DO_TRAIN_WTA;
% artmap_net.DO_TRAIN_ICG = DO_TRAIN_ICG;
% artmap_net.DO_TRAIN_IC = DO_TRAIN_IC;
% artmap_net.DO_TEST_IC = DO_TEST_IC;
% artmap_net.DO_TEST_WTA = DO_TEST_WTA;
% artmap_net.DO_TEST_SCG = DO_TEST_SCG;
% artmap_net.DO_TEST_ICG = DO_TEST_ICG;

% artmap_net.DO_OLD_Tj = DO_OLD_Tj;
% artmap_net.DO_KAPPA_VEC = DO_KAPPA_VEC;

% artmap_net.fail = fail;

%artmap_net.tau_ij = tau_ij; % weights
%artmap_net.tau_ji = tau_ji;
%artmap_net.c = c;           % instance counts
%artmap_net.C = C;           % number of commited nodes
%artmap_net.kappa = kappa;      % Wab
%artmap_net.Delta = Delta;
%artmap_net.rho = rho;
