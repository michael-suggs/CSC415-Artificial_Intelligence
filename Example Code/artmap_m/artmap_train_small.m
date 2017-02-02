function [artmap_net] = artmap_train_small( artmap_net, A, K, verbose, ...
					    n )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% artmap_train_large.m
%
% Description: training an ARTMAP network for a single input
%
% Authors: Suhas Chelian, Norbert Kopco
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs and Defaults Values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% artmap_net = (nothing)
%   The artmap_net to train.
% A = (nothing)
%   The input, complement coded (size = 2M by 1).
% K = (nothing)
%   The output class.
% verbse = = (nothing)
%   A value of 1 will show the beginning and ending of training; a
%   value of 2 will show progress incrementally.
% n = number of training point
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Outputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% artmap_net
%   The artmap_net after training.
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

if ( C == 0 )
% first input
	C = 1;
	y(1) = 1;
	Y(1) = 1;
	sigma_i(1:F0_SIZE,1)=1;

	%// Copy the corresponding output class into variable K
	if ( DO_KAPPA_VEC == 0 )
	  kappa(1)=K;
	else
	  kappa(1,:) = ([1:L] == K);
	end

	%Step 8 - Resonance
	tau_ij(:,1:C) = tau_ij(:,1:C) + ...
	    beta*max(ones(F0_SIZE,1)*y(1,1:C) - tau_ij(:,1:C) - A*ones(1,C),0);
	c(1:C) = c(1:C) + y(1:C);
	tau_ji(1:C,:) = tau_ji(1:C,:) + (ones(C,1)*(beta*max(sigma_i'-A',0)./sigma_i')) .* ...
	    max(Y(1,1:C)'*ones(1,F0_SIZE)-tau_ji(1:C,:),0);
	Delta = [];
	rho = rho_a_bar;

else
  dist_mode = 1; %//revert to distributed mode

  do_reset = 1;
  while do_reset
     %Step 2 - Reset
     do_reset = 0;
     %// F0 -> F2 signal
     S = sum(min(A*ones(1,C),(1-tau_ij(:,1:C))));
     if ( DO_OLD_Tj )
       Theta = sum(tau_ij(:,1:C));
     else
       Theta = sum(tau_ij(:,1:C)) - M;
     end
     
     if ( DO_CBD )
       T(1,1:C) =  S + (1-alpha)*Theta;
     end
     if ( DO_WEBER )
       T(1,1:C) = S ./ (alpha + sum(1-tau_ij(:,1:C)));
       Tu = M/(alpha + 2*M);
     end
     T(1,Delta) = 0;

     %// In F2, Consider nodes whose match is above that of uncommited nodes
     Lambda = find(T>=Tu);
     
     y = zeros(1,C);
     Y = zeros(1,C);
     if ( DO_TRAIN_WTA )
       dist_mode = 0;
     end
     
     %// CAM (F2)
     if(dist_mode)
     % (a) If the network is in distributed mode
       if ( DO_OLD_Tj )
	 alMT = (2-alpha)*M-T;
       else
         alMT = (M-T);
       end
        
       % (i) if M - T >0 for all j belonging to Lambda
       if(all(alMT(Lambda)>0))
	 for j=1:length(Lambda)
	   y(Lambda(j)) = 1/(1+ ...
			     sum(((Lambda~=Lambda(j))*alMT(Lambda(j))./ alMT(Lambda)).^p'));
	 end
       else
       % (ii) Point box case
	 Lambda_pp = find(alMT(Lambda)==0);
	 y(Lambda_pp)=1/length(Lambda_pp);
       end
      
       %// F3 activation
       if ( DO_TRAIN_IC )
	 Y(1:C) = c(1:C).*y(1:C)/sum(c(1:C).*y(1:C));
       else
	 Y(1:C) = y(1:C)/sum(y(1:C));
       end
       
       %// F3 -> F1 signal
       sigma_i(1:F0_SIZE,1) = sum(max(Y(1,1:C)'*ones(1,F0_SIZE) - ...
				  tau_ji(1:C,1:F0_SIZE),0),1)';
     else
        %// (b) If the network is in WTA mode: Only one F2 node, with j=J, is 
	%//		activated
        if ~isempty(Lambda)
	  %// (i) If there is a commited node:
	  J = find(T == max(T(Lambda)));
	  if ( size(J,2) > 1 )
	    J = J(1);
	  end
        else
           % Uncommited node
	  J=C+1;
	  if ( J == MAX_F2_SIZE + 1 ) % add a node
	    aux = zeros(F0_SIZE,1);
	    tau_ij = [tau_ij, aux];
	    tau_ji = [tau_ji; aux'];
	    c = [c 0];
	    if ( DO_KAPPA_VEC == 0 )
	      kappa = [kappa, 0];
	    else
	      aux = zeros(1,L);
	      kappa = [kappa; aux];
	    end
	    MAX_F2_SIZE = MAX_F2_SIZE + 1;
	  end
	  
	  if ( DO_KAPPA_VEC == 0 )
	    kappa(J) = K;
	  else
	    kappa(J,:) = ([1:L] == K);
	  end
	  
	  C=C+1; 

	  if ( verbose >=4 )
	    disp( sprintf( '   New node added, n = %d\n', n ) );
	  end
        end
        
	%// (ii) F2 and F3 activation
        y = ([1:C] == J);
        Y = y;

	%// (iii) F3 -> F1 signal
        sigma_i(1:F0_SIZE,1) = 1-tau_ji(J,:)';

        %// (iv) Add J to the refractory node index set Delta
        Delta = [Delta J];
     end
     
     % Step 3 - Reset or prediction
     %// F1 activation (matching)
     x = min(A, sigma_i);
     
     if(sum(x)/M < rho)
       do_reset = 1;
       dist_mode = 0;
     else
        % Step 4 - Prediction
       if (dist_mode)
	 %// (a) If the network is in distributed mode
	 sigma_k = zeros(L,1);
	 if ( DO_KAPPA_VEC == 0 )
	   for j=1:C
	     sigma_k(kappa(j)) = sigma_k(kappa(j)) + Y(j);
	   end
	 else
	   for j=1:C
	     sigma_k(:) = sigma_k(:) + (kappa(j,:)'*Y(j));
	   end
	 end
	 
	 K_p = find( sigma_k == max(sigma_k) );
	 if ( size(K_p,2) > 1 )
	   K_p = K_p(1);
	 end
       else
	 %// If the network is in WTA mode
	 if ( DO_KAPPA_VEC == 0 )
	   K_p = kappa(J);
	 else
	   sigma_k = kappa(J,:)';
	   [aux,idx] = max( kappa(J,:) );
	   K_p = idx;
	 end
       end
       
       if ( DO_KAPPA_VEC == 1)
	 B = ([1:L] == K);
	 xb = min( B, sigma_k' );
       end
       
       
       % Step 5 - Match tracking or resonance
       if ((DO_KAPPA_VEC == 0 & (K_p ~= K) ) |...
	   (DO_KAPPA_VEC == 1 & (sum(xb) < rho_ab) ))
         % Step 6 - Match tracking
	 rho = 1/M * sum(x) + epsilon;
	 dist_mode = 0; %//revert to WTA
	 do_reset = 1;
       else
         % Correct prediction
	 if dist_mode
           % Step 7 - Credit assignment
	   %// F2 blackout of incorretly predicting nodes
	   
	   if ( DO_KAPPA_VEC == 0 )
	     y(1:C) = y(1:C) .*(kappa(1:C) == K);
	   else
	     for j = 1:C
	       % Digital Blackout
	       [aux,idx] = max(kappa(j,:));
	       y(j) = y(j) * aux * (idx == K);

%	       if ( idx == K )
%		 y(j) = y(j);
%	       else
%		 y(j) = 0;
%	       end
	     
	       % Analog Blackout
	       %deg_corr = 1- (1-kappa(j,K)); % degree of correctness:
	       %                              % 1 for totally correct,
	       %                              % 0 for totally wrong
	       %                              % (assuming sum(
	       %                              % kappa(j,:) ) = 1 )
	       %y(j) = y(j) * deg_corr;
	     end
	     
	   end
	
	   %// F2 renormalization
	   y(1:C) = y(1:C) / sum(y(1:C));

	   %// F3 renormalization
	   if ( DO_TRAIN_IC )
	     Y(1:C) = c(1:C).*y(1:C) / sum(c(1:C).*y(1:C));
	   else
	     Y(1:C) = y(1:C) / sum( y(1:C) );
	   end

	   %// F3 -> F1 signal
	   sigma_i(1:F0_SIZE,1) = sum(max(Y(1,1:C)'*ones(1,F0_SIZE) - tau_ji(1:C,1:F0_SIZE),0),1)';
	 end
           
         % Step 8 - Resonance (another copy above)
	 tau_ij(:,1:C) = tau_ij(:,1:C) + ...
	     beta*max(ones(F0_SIZE,1)*y(1,1:C) - tau_ij(:,1:C) - A*ones(1,C),0);
	 aux = (ones(C,1)*(beta*max(sigma_i'-A',0))) .* ...
	       max(Y(1,1:C)'*ones(1,F0_SIZE)-tau_ji(1:C,:),0);
	 aux = aux./(ones(C,1)*(sigma_i+(sigma_i==0))');
	 tau_ji(1:C,:) = tau_ji(1:C,:) + aux;
	 
	 c(1:C) = c(1:C) + y(1:C);
	 
	 % Allow slow learning of Mapfield only in WTA mode.  Note
         % that the tau and c update rules need to be fixed too.
 	 if ( DO_TRAIN_WTA )
 	   if ( DO_KAPPA_VEC == 0 )
 	     kappa(J) = K;
 	   else
 	     kappa(J,:) = (1-beta)*kappa(J,:) + beta*B;
 	   end
 	 end
	 
	 Delta = [];
	 rho = rho_a_bar;
       end
       
     end
  end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pack
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

artmap_net.type = MAPTYPE;
artmap_net.M = M;
artmap_net.L = L;
artmap_net.EPOCHS = EPOCHS;
artmap_net.MAX_F2_SIZE = MAX_F2_SIZE;

artmap_net.TRAIN_N = TRAIN_N;
artmap_net.input = input;
artmap_net.output = output;
artmap_net.TEST_N = TEST_N;
artmap_net.te_input = te_input;
artmap_net.te_output = te_output;

artmap_net.alpha = alpha;
artmap_net.p = p;
artmap_net.beta = beta;
artmap_net.epsilon = epsilon;
artmap_net.rho_a_bar = rho_a_bar;
artmap_net.rho_ab = rho_ab;
artmap_net.Tu = Tu;

artmap_net.F0_SIZE = M*2;
artmap_net.F2_SIZE = MAX_F2_SIZE;

artmap_net.DO_WEBER = DO_WEBER;
artmap_net.DO_CBD = DO_CBD;
artmap_net.DO_TRAIN_WTA = DO_TRAIN_WTA;
artmap_net.DO_TRAIN_ICG = DO_TRAIN_ICG;
artmap_net.DO_TRAIN_IC = DO_TRAIN_IC;
artmap_net.DO_TEST_IC = DO_TEST_IC;
artmap_net.DO_TEST_WTA = DO_TEST_WTA;
artmap_net.DO_TEST_SCG = DO_TEST_SCG;
artmap_net.DO_TEST_ICG = DO_TEST_ICG;

artmap_net.DO_OLD_Tj = DO_OLD_Tj;
artmap_net.DO_KAPPA_VEC = DO_KAPPA_VEC;

artmap_net.tau_ij = tau_ij; % weights
artmap_net.tau_ji = tau_ji;
artmap_net.c = c;           % instance counts
artmap_net.C = C;           % number of commited nodes
artmap_net.kappa = kappa;      % Wab
artmap_net.Delta = Delta;
artmap_net.rho = rho;
