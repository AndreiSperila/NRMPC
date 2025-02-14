%% Perform cleanup before running the script

clear; clc; close all;

% References:

% [1] Şerban Sabău, Cristian Oară, Sean Warnick, and Ali Jadbabaie. 
%     Optimal Distributed Control for Platooning via Sparse Coprime
%     Factorizations. IEEE Transactions on Automatic Control, 
%     62(1):305–320, 2017.

% [2] Andrei Sperilă, Alessio Iovine, Sorin Olaru, and Patrick Panciatici. 
%     Network-Realized Model Predictive Control – Part I: NRF-Enabled 
%     Closed-loop Decomposition, 2024.
%     Available Online: https://arxiv.org/abs/XXXX.YYYYY.

% [3] Andrei Sperilă, Alessio Iovine, Sorin Olaru, and Patrick Panciatici. 
%     Network-Realized Model Predictive Control – Part II: Distributed 
%     Constraint Management, 2024.
%     Available Online: https://arxiv.org/abs/XXXX.YYYYY.

% [4] Martin Herceg, Michal Kvasnica, Colin Jones, and Manfred Morari. 
%     Multi-Parametric Toolbox 3.0. In Proc. of the European Control 
%     Conference, 2013

% [5] MOSEK ApS. The MOSEK optimization toolbox for MATLAB manual.
%     Version 10.2, 2024.

% [6] Johan Löfberg. YALMIP: A Toolbox for Modeling and Optimization in
%     MATLAB, In Proc. of the CACSD Conference, 2004.

%% Setup the dynamical models of the platoon's cars

% For more information on the origin of these dynamics, see [1].

d_t = 0.1; % Sampling time
h = 5; % Time headway
N = 10; % Number of cars in the platoon
M = ones(N,1); % Mass of each car
tau = 0.1*ones(N,1); % Time constant of each car's actuator
sigma = ones(N,1); % Constant given by each car's geometric configuration
car_len_prev = [0 5*ones(1,N-1)]; % Length of the car located directly in
% front of each car (the 0th car is virtual, so its length is 0).

% Define the double-integrator part of each car's dynamics

A_i = [0 1;0 0];
B_i = [0;1];
C   = [1 0]; % Position-selecting matrix
Ch  = [1 h]; % Matrix to select the combined state information for feedback

% Define the actuator dynamics, and then proceed to discretize them along
% with the previously defined double-integrators. In addition to this, we
% assume that the actuators' states cannot be directly measured. However,
% we do assume that the accelerations of the cars may be measured and that
% the actuators' models can be used to synthesize Luenberger state
% observers. As will be explained in the sequel, the platoon's actuators
% are reset to default when booting up the control system (which corresponds,
% mathematically, to their states being set to 0), so the state estimates
% provided by these observers is reliable and all inaccuracies (given by
% model mismatch and accelerometer noise) can be incorporated into the
% state noise variable employed in the sequel.

G_act_c  = cell(1,N);
G_act    = cell(1,N);
G_car_c  = cell(1,N);
G_car    = cell(1,N);
act_est  = cell(1,N);
L_act    = zeros(1,N); % Since the actuator dynamics are stable, we take
% observer gain as the zero matrix, which also desensitizes the observer to
% accelerometer noise.
for i=1:N
    G_act_c{i} = ss(-1/tau(i),1/tau(i), ...
                 (sigma(i)*tau(i)-1)/(tau(i)*M(i)),1/(tau(i)*M(i)));
    G_car_c{i} = ss(A_i,B_i,eye(2),0) * G_act_c{i};
    G_act{i}   = c2d(G_act_c{i}, d_t, 'zoh');
    G_car{i}   = c2d(G_car_c{i}, d_t, 'zoh');
    act_est{i} = ss(G_act_c{i}.a+L_act(i)*G_act_c{i}.c,...
                   [-L_act(i) G_act_c{i}.b+L_act(i)*G_act_c{i}.d],1,[0 0]);
    act_est{i} = c2d(act_est{i},d_t,'zoh');
end

%% For the platoon's full state dynamics

% In order to obtain the recurrent equation from (50) in [3], we first
% merge all the car's dynamics into a single state-space system and we
% apply the state equivalence transformation given in Section 7.1.2 of [3]

[n,m] = size(G_car{1}.b);
T_state = eye(n*N+N+1);
A_p = eye(N+1);
B_p = [1; zeros(N,1)];
C_pn = zeros(1,N+1);
for i = 1:N
    A_p = blkdiag(A_p,G_car{i}.a);
    B_p = blkdiag(B_p,G_car{i}.b);
    C_pn = blkdiag(C_pn,[Ch 0]);
    if i > 1
        T_state(N+1+n*(i-1)+1,N+1+n*(i-2)+1) = -1;
    else
        T_state(N+2,1) = -1;
    end
    T_state(N+1+n*(i-1)+1,1+i) = 1;
end
A_pn = T_state * (A_p / T_state);
B_pn = T_state * B_p;
G_pn = ss(A_pn,B_pn,C_pn(2:end,:),0,d_t); % The platoon's model in which
% the outputs are the feedback signals of the first layer subcontrollers.

%% Close the loop formed by the platoon and the first layer subcontrollers

load('NRF_implem.mat') % Load the NRF-based implementations of the first
% layer subcontrollers. As discussed in [3], the synthesis procedure of
% these control laws has already been investigated from both practical [1]
% and theoretical [2] standpoints. Thus, the example discussed here focuses
% on the synthesis procedure of the second layer, with a particular
% emphasis on the set-theoretical approach discussed in Remark 7.2 of [3].

scal_mat = blkdiag(eye(N*m),-eye(N)); % Switch from the negative feedback
% setting in [1], to the positive one in [2]-[3].

% Form the positive-feedback "global" controller of the first layer.
K_D_d_gen = ss(K_D_sliced_d{1}.a,K_D_sliced_d{1}.b*scal_mat,...
               K_D_sliced_d{1}.c,K_D_sliced_d{1}.d*scal_mat,d_t);
for i = 2:N
    K_D_d_gen = ss(blkdiag(K_D_d_gen.a,K_D_sliced_d{i}.a),...
                   [K_D_d_gen.b; K_D_sliced_d{i}.b*scal_mat],...
                   blkdiag(K_D_d_gen.c,K_D_sliced_d{i}.c),...
                   [K_D_d_gen.d; K_D_sliced_d{i}.d*scal_mat],d_t);
end

% Form the closed-loop dynamics given in (52) from [3].
full_cl_sys = ss([G_pn.a G_pn.b(:,2:end)*K_D_d_gen.c;...
                  K_D_d_gen.b(:,N*m+1:end)*G_pn.c...
                  K_D_d_gen.a+K_D_d_gen.b(:,1:N*m)*K_D_d_gen.c],...
                  blkdiag(G_pn.b, [K_D_d_gen.b(:,1:N*m)...
                  K_D_d_gen.b(:,N*m+1:end)*G_pn.c(:,N+2:end)]),...
                  blkdiag(eye(size(G_pn.a)), eye(size(K_D_d_gen.a))),...
                  0,d_t);

% Validate the block-diagonal structure of the state matrix in (52)
% from [3], along with the stability of the relevant block-partition.
norm(full_cl_sys.a(1:N+1,N+2:end))
norm(full_cl_sys.a(N+2:end,1:N+1))
max(abs(eig(full_cl_sys.a(N+2:end,N+2:end))))

%% Partition the first layer closed-loop system into areas

ns = n*N;
ms = m*N;

% Seelct the reduced dynamics in accordance with the arguments presented in
% Remark 7.1 of [3].
red_cl_sys = ss(full_cl_sys.a(N+2:end,N+2:end),...
                full_cl_sys.b(N+2:end,:), blkdiag(eye(ns),K_D_d_gen.c),...
                0,d_t);

area_sys = cell(1,N);
n_w      = zeros(1,N);
for i=1:N
    area_sys{i} = red_cl_sys([(i-1)*n+1:i*n ns+i],...
                              [1+2*ms+n*(i-1)+1 1+2*ms+n*(i-1)+2 1+i]);
    area_sys{i} = ss(area_sys{i},'min'); % One of the main advantages of
    % proposed approach is the fact that, since we manipulate the area
    % models dealing with the system's forced response, a minimal (and, 
    % therefore, more computationally efficient) realization is just as
    % suitable as the original one.
    n_w(i) = length(K_D_sliced_d{i}.a);
end

%% Set up the box constraints of the closed-loop system's variables

amp_n_x = 2e-2; % Although we assume that the sensors employed by the
% platoon's cars are affected by an additive noise which is upper-bounded 
% in absolute value by 1e-2, note that the car interdistance is computed
% by combining two position sensor measurements. Thus, the noise affecting
% it is effectively doubled (in the worst case situation) and we adopt this
% doubled bound for all other measurements, for the sake of convenience 
% (while accepting a small amount of conservatism). For simulation purposes,
% this variable will be reset to 1e-2, after all operations related to set
% computation have been successfully completed.
amp_n_u_f = 2e-2; % Noise associated with first layer subcontroller 
% communication.
amp_n_w = amp_n_u_f; % Since each NRF subcontroller has one state, which is
% also its output, the same disturbance amplitude as above is applicable.
amp_n_u_s1 = 1e-2; % Numerical perturbations associated with the output of
% the second layer subcontrollers.
amp_n_u_s2 = 1e-2; % Same as above.

U_lims = 10 * ones(1,N); % Control signal limits to each car in the platoon.
amp_a = U_lims; % Limits on the state of the actuator, taken to be the same
% as that of the control signal, due to the first-order system nature of
% the actuator state dynamics (as defined earlier in this script).

% We now set limits on both car interdistance (which take into account each
% vehicle's length) and absolute car speed.

interdist_max = 0;
interdist_min = -360;
speed_max = 36;
speed_min =  0;

X_lims = zeros(3,N);
for i=1:N
    X_lims(:,i) = [max(abs(interdist_max), abs(interdist_min));...
                   max(abs(speed_max), abs(speed_min));...
                   U_lims(i)];
end

% We choose the range of values for the two sets of control signals
% produced by the second layer: those sent to the NRF subcontrollers and
% those sent directly to the platoon, respectively. The former ignore the
% identically zero input channel of the first layer subcontroller, and
% their amplitude is chosen sufficiently large so as to invert the sign of
% the NRF's feedback information. The signals sent to the platoon are
% chosen as half of the available range, with the first layer's
% subcontrollers being assigned the other half, in what amounts to a
% balanced collaborative effort between the two control layers.
U_s1_lims = 2 * diag([1 1 0]) * X_lims;
U_s2_lims = 0.5 * U_lims;
U_f_lims = U_lims - (U_s2_lims + amp_n_u_s2*ones(1,ms));
W_lims = U_f_lims;

%% Set up the variables associated with each area

% We begin with the polyhedral sets, in H-representation.

H_X = cell(1,N);
w_X = cell(1,N);
H_U = cell(1,N);
w_U = cell(1,N);
H_D = cell(1,N);
w_D = cell(1,N);
H_W = cell(1,N);
w_W = cell(1,N);
H_N = cell(1,N);
w_N = cell(1,N);
H_J = cell(1,N);
w_J = cell(1,N);

X_set = cell(1,N); % Current car state constaint sets.
U_set = cell(1,N); % Second layer command constraint sets.
D_set = cell(1,N); % Exogenous disturbance bounding sets.
W_set = cell(1,N); % Previous car information bounding sets.
N_set = cell(1,N); % Measurement noise sets.
C_set = cell(1,N); % Robust control invariant sets.
J_set = cell(1,N); % Sets employed in the MPC subcontrollers in order to
% constrain the one-step predicted values of the noise-affected states.

% We also set up the state-space realizations shown in (51a)-(51b) from [3].

A_area = cell(1,N);
B_area = cell(1,N);
A_area_p = cell(1,N);
B_area_d = cell(1,N);

%% Compute the initialized sets and realizations

% All set computations in this example have been carried out via MPT3 [4],
% through the use of the MOSEK [5] solver.

ctl_shift = 0;
for i=1:N
    % Select the representations given in (51a)-(51b) of [3] from the
    % closed-loop system between the platoon and the first layer.
    A_area{i} = red_cl_sys.a([(i-1)*n+1:i*n ns+ctl_shift+1:ns+ctl_shift+n_w(i)],...
                             [(i-1)*n+1:i*n ns+ctl_shift+1:ns+ctl_shift+n_w(i)]);
    B_area{i} = red_cl_sys.b([(i-1)*n+1:i*n ns+ctl_shift+1:ns+ctl_shift+n_w(i)],...
                             [1+i 1+2*ms+(i-1)*n+1:1+2*ms+(i-1)*n+2]);
    if i>1
        A_area_p{i} = red_cl_sys.a([(i-1)*n+1:i*n ns+ctl_shift+1:ns+ctl_shift+n_w(i)],...
                                   [(i-2)*n+1:(i-1)*n ns+ctl_shift-n_w(i-1)+1:ns+ctl_shift]);
        B_area_d{i} = red_cl_sys.b([(i-1)*n+1:i*n ns+ctl_shift+1:ns+ctl_shift+n_w(i)],...
                                   [i 1+i 1+ms+(i-1) 1+2*ms+(i-1)*n+1:1+2*ms+(i-1)*n+2]);
    else
        B_area_d{1} = red_cl_sys.b([1:n ns+1:ns+n_w(1)],...
                                   [2 2*ms+2:2*ms+1+n-1]);
        B_area_p    = red_cl_sys.b([1:n ns+1:ns+n_w(1)],1);
    end
    ctl_shift = ctl_shift + n_w(i);

    % Set up each car's state constraints and add the additional
    % restriction which forces the one-step positional increment to be
    % within what can be achieved by the each car's speed limits, assuming
    % worst case actuation by the second layer subcontroller. By doing so,
    % the car directly behind it in the platoon may always match the
    % positional increment of the one in front, while also staying within
    % the prescribed speed limits.
    H_X{i} = blkdiag(eye(n),eye(n_w(i)));
    H_X{i} = [H_X{i}; -H_X{i}; [0 A_area{i}(1,2:4)]; [0 -A_area{i}(1,2:4)]];
    dp_max = speed_max*d_t - ((U_s2_lims(i)+amp_n_u_s2) * abs(G_car{i}.b(1)));
    dp_min = speed_min*d_t + ((U_s2_lims(i)+amp_n_u_s2) * abs(G_car{i}.b(1)));
    w_X{i} = [interdist_max; speed_max; U_lims(1); W_lims(i)];
    w_X{i} = [w_X{i}; -interdist_min; -speed_min; U_lims(1); W_lims(i);...
              dp_max; -dp_min];
    X_set{i} = Polyhedron(H_X{i},w_X{i});

    % set up platoon and first layer state measurement noise sets.
    H_N{i} = blkdiag(eye(n),eye(n_w(i)));
    H_N{i} = [H_N{i}; -H_N{i}];
    w_N{i} = [amp_n_x*ones(n,1); amp_n_w*ones(n_w(i),1)];
    w_N{i} = [w_N{i}; w_N{i}];
    N_set{i} = Polyhedron(H_N{i},w_N{i});
    
    % Set up the second layer command constraints.
    H_U{i} = [eye(m+n-1)];
    H_U{i} = [H_U{i}; -H_U{i}];
    w_U{i} = [U_s2_lims(:,i);U_s1_lims(1:end-1,i)];
    w_U{i} = [w_U{i}; w_U{i}];
    U_set{i} = Polyhedron(H_U{i},w_U{i});

    % Handle separately the sets which bound disturbance and the 
    % information received from the previous car, in accordance with the
    % two models given in (51a)-(51b) from [3].
    if i>1
        H_D{i} = [eye(2*m+1+(n-1))];
        H_D{i} = [H_D{i}; -H_D{i}];
        w_D{i} = [U_s2_lims(i-1)+amp_n_u_s2; amp_n_u_s2; amp_n_u_f;...
                 (amp_n_x + amp_n_u_s1)*ones(n-1,1);];
        w_D{i} = [w_D{i}; w_D{i}];
        D_set{i} = B_area_d{i} * Polyhedron(H_D{i},w_D{i});
        D_set{i} = D_set{i}.minHRep;

        W_set{i} = A_area_p{i} * (X_set{i-1} + N_set{i-1});
        W_set{i} = W_set{i}.minHRep;
    else
        H_D{1} = [eye(m+n-1)];
        H_D{1} = [H_D{1}; -H_D{1}];
        w_D{1} = [amp_n_u_s2; (amp_n_x + amp_n_u_s1)*ones(n-1,1)];
        w_D{1} = [w_D{1}; w_D{1}];
        D_set{1} = B_area_d{1} * Polyhedron(H_D{1},w_D{1});
        D_set{1} = D_set{1}.minHRep;

        H_W{1} = [1; -1];
        w_W{1} = [speed_max*d_t; -speed_min*d_t];
        W_set{1} = B_area_p * Polyhedron(H_W{1},w_W{1});
        W_set{1} = W_set{1}.minHRep;
    end

end

%% Compute the robust control invariant sets

max_iter = 10; % Numerical test have shown that, for this numerical example
% and the parameters chosen above, the shape of the robust control
% invariant sets suffers only insignificant changes after the 10th
% iteration, while the number of halfspaces needed to represent these
% subsequent polyhedrons increases dramatically. When projected onto the
% first two coordinates, the reason becomes clear: the sequence of 2D
% projections converges to a shape with very rounded curves, which requires
% a large number of inequalities in order to be fully represented.

for i=1:N

    % Compute the inverse of the state matrix and perform the
    % initialization via the adapted state constraint set.
    iter = 1;
    Ai = linsolve(A_area{i},eye(size(A_area{i})));
    dist_set_1 = (-A_area{i}) * N_set{i};
    dist_set_1 = dist_set_1.minHRep;
    dist_set_1 = D_set{i} + dist_set_1;
    dist_set_1 = dist_set_1.minHRep;
    J_set{i} = X_set{i} - dist_set_1; 
    J_set{i} = J_set{i}.minHRep;
    dist_set_2 = A_area{i} * N_set{i};
    dist_set_2 = dist_set_2.minHRep;
    dist_set_2 = W_set{i} + dist_set_2;
    dist_set_2 = dist_set_2.minHRep;
    dist_set_2 = A_area{i} * dist_set_1 + dist_set_2;
    dist_set_2 = dist_set_2.minHRep;

    % The real state dynamic is 
    % x_i[k+1] = A_area{i} * x_i[k] + _area{i} * u_i[k] + w_i[k] + d_i[k],
    % where w_i[k] in W_set{i} is the known exogenous disturbance and d_i[k]
    % in D_set{i} is the unknown one. Each MPC subcontroller considers
    % x_hat_i[k+1] = A_area{i} * x_hat_i[k] + B_area{i} * u_i[k] + w_i[k]
    % as the plant model, where the new state vector can be written as
    % x_hat_i[k] = x_i[k] + q_i[k], for some q_i[k] in N_set{i}. We compute 
    % the maximal robust control invariant set (MRICS) J_set{i} for
    % x_hat_i[k] as below, and we assume that x_i[k] is initialized in
    % C_set{i} = J_set{i} + D_set{i} + ((-A_area{i}) * N_set{i}).
    % Then, by the construction of J_set{i}, it is always possible to force
    % x_hat[k+1] to stay inside the aforementioned set, from which we get that
    % x_i[k+1] = x_hat_i[k+1] + d_i[k] + (-A_area{i} * q_i[k]) is in C_set{i}.

    % The computation of the MRICS sets is the standard one, encountered in 
    % most textbooks on set-theoretical methods.
    while iter <= max_iter

        disp(iter);
        
        temp_set = J_set{i} + ((-B_area{i}) * U_set{i});
        temp_set = temp_set.minHRep;
        temp_set = temp_set - dist_set_2;
        temp_set = temp_set.minHRep;
        temp_set = Ai * temp_set;
        temp_set = temp_set.minHRep;
        temp_set = intersect(temp_set,J_set{i}.minHRep);
        temp_set = temp_set.minHRep;

        if temp_set.isEmptySet
            error('MRCIS is empty!');
        else
            if temp_set == J_set{i}
                break;
            else
                J_set{i} = temp_set;
                iter = iter + 1; 
            end
        end
        
    end

    temp_set = J_set{i} + dist_set_1;
    C_set{i} = temp_set.minHRep;
    
    if iter > max_iter
        fprintf('Maximum iteration limit reached on car %d!\n',i);
    else
        fprintf('Done with car %d!\n',i);
    end

end

clear temp_set dist_set_1 dist_set_2

save('post_set_comp.mat')

%% Simulink scheme initialization

% We now reset the car state measurement noise bounds for simulation
% purposes, in accordance with the comments made above.
amp_n_x = 1e-2;

% We select the weighting matrices for the headway-based interdistance and
% for the second layer command values, respectively, as in (58) from [3]. 
% Given that we want the second layer to intervene as little as possible in
% the functioning of the first layer's closed-loop system (only when 
% necessary, in order to guarantee constraint satisfaction), we select
% comparatively large weights for command values.
Q_cost = cell(1,N);
R_cost = cell(1,N);
for i=1:N
    Q_cost{i} = 1e-9;
    R_cost{i} = eye(m+n-1);
    H_J{i} = J_set{i}.A;
    w_J{i} = J_set{i}.b;
end

% We introduce the following two variables, which will be employed in the
% Simulink scheme to compute the reference signal (the 0th car's position).
pos_ref = h*10;
spd_ref = 10;

% Finally, we load the initial conditions of the platoon's realization, in
% terms of car position and speed. Recall that all actuators are reset at
% initialization, with their states being set to 0.
load('init_scatter.mat');

%% Check the feasibility of the initial conditions

% Similarly to the actuator states, each NRF subcontroller's state is
% initialized in 0. Then, each optimization problem designating an
% MPC-based subcontroller is tested for feasibility. Given the theoretical
% developments obtained in [3], if each of these is feasible, then they are
% all guaranteed to be recursively feasible for the considered scenario.
% For implementation purposes, the following optimization problems (along 
% with the ones solved in the Simulink scheme) have been carried out using
% the MOSEK [5] solver via YALMIP [6].

if C_set{1}.contains([x0_c(1,1)-pos_ref+car_len_prev(1); x0_c(2:3,1); 0])
    car_MPC_leader([x0_c(1,1)-pos_ref+car_len_prev(1); x0_c(2:3,1); 0],...
                    3, [3; 1], 0)
else
    error('Initial conditions do not guarantee recursive feasibility');
end
pause()

if C_set{2}.contains([x0_c(1,2)-x0_c(1,1)+car_len_prev(1); x0_c(2:3,1); 0])
    car_MPC_follower([x0_c(1,2)-x0_c(1,1)+car_len_prev(1); x0_c(2:3,1); 0],...
                     [x0_c(1,1)-pos_ref+car_len_prev(1); x0_c(2:3,1); 0],...
                     [2; 3; 1],0)
else
    error('Initial conditions do not guarantee recursive feasibility');
end
pause()

for i=3:N
    if C_set{i}.contains([x0_c(1,2)-x0_c(1,1)+car_len_prev(1); x0_c(2:3,1); 0])
        car_MPC_follower([x0_c(1,i)-x0_c(1,i-1)+car_len_prev(i); x0_c(2:3,i); 0],...
                         [x0_c(1,i-1)-x0_c(1,i-2)+car_len_prev(i-1);...
                         x0_c(2:3,i-1); 0],[i; 3; 1],0)
    else
        error('Initial conditions do not guarantee recursive feasibility');
    end
    if i < N
        pause()
    end
end

clear ans

%% Save the results of the design phase for future use

save('platoon_implementation_variables')
