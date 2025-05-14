%% This code is modeling the individual farmer maximize the profit subject to the pollutant cumulation by
% choosing the mulch decay rate. Given the optimal disposal strategy is to
% dispose of all the pre-tilled mulches



function Z1Individual5year
%% FORMULATION
demosetup(mfilename)

% Model parameters
a = [18000 1 0.1 0.005 5 0];
b = [0 0 1237.56 -0.5];
nu =[18621.39 15];
%ita 1 was 3.20, but we added 0.05 of residue which is A base level of residue from PE (or BDM for that matter) whether or not the mulch goes to a landfill
ita = [3.25 0.5 0.1 3.2 -0.5];
delta = 0.9;
w = 0.052490137;
p = 1.01;
z = 1;

% Model structure
model.func = @func;                             % model functions
model.params = {a,b,nu,ita,z,w,p};              % function parameters
model.discount = delta;                         % discount factor
model.ds = 1;                                   % dimension of continuous state
model.dx = 1;                                   % dimension of continuous action
model.ni = 0;                                   % number of discrete states
model.nj = 0;                                   % number of discrete actions

% Approximation structure
n    = 100;                                     	% number of collocation nodes
smin = 0;                                       	% minimum state
smax = 50;                                          % maximum state
basis = fundefn('spli',n,smin,smax);                % basis functions

 
%% SOLUTION
  
% Steady-state
format long
qstar =max(0,((nu(2)+w*(-b(2)+b(4)*z)+a(2)*p)*(1-delta*ita(3))*(1-ita(3))+(a(6)...
    *(1-delta*ita(3))+delta*a(5)*(-ita(2)-ita(5)*z))*p*(ita(1)-ita(4)*z)+...
    a(3)*delta*(-ita(2)-ita(5)*z)*p*(1-ita(3)))/...
    ((-a(4)*(1-ita(3)*delta)+a(6)*delta*(ita(2)+ita(5)*z))*p*(1-ita(3))+(-a(6)*(1-delta*ita(3))...
    +(a(5)*delta*(ita(2)+ita(5)*z))*p*(-ita(2)-ita(5)*z)))); 	% choice variable
sstar = (ita(1)-ita(2)*qstar-ita(4)*z-ita(5)*z*qstar)/(1-ita(3));           % state variable
lstar = p*(-a(3)-a(5)*sstar-a(6)*qstar)/(-delta*ita(3)+1);                            % shadow price
pistar = (p*(a(1)-a(2)*qstar-a(3)*sstar-0.5*a(4)*qstar.^2-0.5*a(5)*sstar.^2-a(6)*qstar.*sstar)...
        -nu(1)-nu(2)*qstar-w*(b(1)-b(2)*qstar+b(3)*z+b(4)*z*qstar))/(1-delta); % The present value of life time profit givern the optimal choice
fprintf('Steady States\n') 
fprintf('   pollutant Stock         %5.2f\n'  ,sstar)
fprintf('   decay rate       %5.2f\n'  ,qstar)
fprintf('   Shadow Price  %5.2f\n'  ,lstar)
fprintf('   lifetime profit  %5.2f\n'  ,pistar)

% check the condition that the optimal z=1, check value>0
qstar1 =((nu(2)+w*(-b(2)+b(4)*z)+a(2)*p)*(1-delta*ita(3))*(1-ita(3))+(a(6)...
    *(1-delta*ita(3))+delta*a(5)*(-ita(2)-ita(5)*z))*p*(ita(1)-ita(4)*z)+...
    a(3)*delta*(-ita(2)-ita(5)*z)*p*(1-ita(3)))/...
    ((-a(4)*(1-ita(3)*delta)+a(6)*delta*(ita(2)+ita(5)*z))*p*(1-ita(3))+(-a(6)*(1-delta*ita(3))...
    +(a(5)*delta*(ita(2)+ita(5)*z))*p*(-ita(2)-ita(5)*z))); % choice variable without boundary
sstar1 = (ita(1)-ita(2)*qstar1-ita(4)*z-ita(5)*z*qstar1)/(1-ita(3));           % state variable without boundary
lstar1 = p*(-a(3)-a(5)*sstar1-a(6)*qstar1)/(-delta*ita(3)+1);                            % shadow price without boundary

check = -w*(b(3)+b(4)*qstar1)+delta*(-ita(4)-ita(5)*qstar1)*lstar1;
fprintf('check the condition that the optimal z=1, check value>0 %5.2f\n' , check)
% Chekc the condition that the FOC=0
check1=-a(2)*p-a(4)*qstar1*p-a(6)*sstar1*p-nu(2)+w*b(2)-b(4)*z*w+delta*lstar1*(-ita(2)-ita(5)*z);
fprintf('Chekc the condition that the FOC~=0 %5.2f\n' , check1)
% Check model derivatives
dpcheck(model,sstar,qstar);
% Solve collocation equation
[c,s,v,q,resid] = dpsolve(model,basis);

%% SIMULATION

% Simulation parameters
nper = 1000;                              % number of periods simulated

% Initialize simulation
sinit = smin;                           % agent possesses minimal wealth

% Simulate model
[ssim,qsim] = dpsimul(model,basis,nper,sinit,[],s,v,q);


% Calculate the 2.5th percentile
lower_bound = prctile(ssim, 2.5);

% Calculate the 97.5th percentile
upper_bound = prctile(ssim, 97.5);

% Display the results
disp(['2.5th percentile of pollutant stock: ', num2str(lower_bound)]);
disp(['97.5th percentile of pollutant stock: ', num2str(upper_bound)]);

% Calculate the 2.5th percentile
lower_bound_q = prctile(qsim, 2.5);

% Calculate the 97.5th percentile
upper_bound_q = prctile(qsim, 97.5);

% Display the results
disp(['2.5th percentile of decay rate: ', num2str(lower_bound_q)]);
disp(['97.5th percentile of decay rate: ', num2str(upper_bound_q)]);
% Plot simulated state and policy paths
figure
hold on
plot(0:nper-1,ssim,'color',[0,0,0])
plot(0:nper-1,qsim,'color',[0.6,0.6,0.6])
ylim([0 0.2])
xlabel('Period')
ylabel('Simulated Stock and Degradation Rate')

% ...plot steady-state
plothdash([],sstar,[0,0,0])
plothdash([],qstar,[0.6,0.6,0.6])
plottext([],sstar,'$s^*$ ')
plottext([],qstar,'$\delta^*$')
legend('Pollutant Stock','Degradation Rate','$s^*$steady state','$\delta^*$steady state')



%% DPSOLVE FUNCTION FILE


function [out1,out2,out3] = func(flag,s,q,~,~,~,a,b,nu,ita,z,w,p)

switch flag
  case 'b'      % bounds
    out1 = zeros(size(s));
    out2 = ones(size(s));
    out3 = [];
  case 'f'      % Profit
    out1 = p*(a(1)-a(2)*q-a(3)*s-0.5*a(4)*q.^2-0.5*a(5)*s.^2-a(6)*q.*s)...
        -nu(1)-nu(2)*q-w*(b(1)-b(2)*q+b(3)*z+b(4)*z*q);
    out2 = -a(2)*p-a(4)*q*p-a(6)*s*p-nu(2)+w*b(2)-b(4)*z*w;
    out3 =  -a(4)*p*ones(size(s));
  case 'g'      % Motion of state variable
    out1 = ita(1)-ita(2)*q+ita(3)*s-ita(4)*z-ita(5)*q*z;
    out2 = (-ita(2)-ita(5)*z)*ones(size(s));
    out3 = zeros(size(s));
end