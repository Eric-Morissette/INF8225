fprintf('TP1 - Question 2:\n\n');

fprintf(' [C]   [T]\n');
fprintf('   \\   /|\n');
fprintf('    [A] |\n');
fprintf('   /   \\|\n');
fprintf(' [M]   [J]\n\n');

C = 1; T = 2; A = 3; M = 4; J = 5;

names = cell(1,5);
names{C} = 'Cambriolage';
names{T} = 'Tremblement';
names{A} = 'Alarme';
names{M} = 'MarieAppelle';
names{J} = 'JeanAppelle';

dgm = zeros(5,5);
dgm(C, A) = 1;
dgm(T, A) = 1;
dgm(T, J) = 1;
dgm(A, M) = 1;
dgm(A, J) = 1;

CPDs{C} = tabularCpdCreate(reshape([0.999 0.001], 2, 1));
CPDs{T} = tabularCpdCreate(reshape([0.998 0.002], 2, 1));
CPDs{A} = tabularCpdCreate(reshape([0.999, 0.06, 0.71, 0.05, 0.001, 0.94, 0.29, 0.95], 2, 2, 2));
CPDs{M} = tabularCpdCreate(reshape([0.95 0.1 0.05 0.9], 2, 2));
CPDs{J} = tabularCpdCreate(reshape([0.999 0.999 0.5 0.95 0.001 0.001 0.5 0.05], 2, 2, 2));

dgm = dgmCreate(dgm, CPDs, 'nodenames', names, 'infEngine', 'jtree');
joint = dgmInferQuery(dgm, [C, T, A, M, J]);



fprintf('(C) - Marginales Conditionnelles\n');
clampled = sparsevec([M J], [2 1], 5);
C_Sachant_M_J = tabularFactorCondition(joint, C, clampled);
fprintf('C Sachant !M  J: p(C=1|M=1,J=0)=%f\n', C_Sachant_M_J.T(2));

clampled = sparsevec([M J], [1 2], 5);
C_Sachant_M_J = tabularFactorCondition(joint, C, clampled);
fprintf('C Sachant !M  J: p(C=1|M=0,J=1)=%f\n', C_Sachant_M_J.T(2));

clampled = sparsevec([M J], [2 2], 5);
C_Sachant_M_J = tabularFactorCondition(joint, C, clampled);
fprintf('C Sachant  M  J: p(C=1|M=1,J=1)=%f\n', C_Sachant_M_J.T(2));

clampled = sparsevec([M J], [1 1], 5);
C_Sachant_M_J = tabularFactorCondition(joint, C, clampled);
fprintf('C Sachant !M !J: p(C=1|M=0,J=0)=%f\n', C_Sachant_M_J.T(2));

clampled = sparsevec(M, 2, 5);
C_Sachant_M = tabularFactorCondition(joint, C, clampled);
fprintf('C Sachant  M   : p(C=1|M=1)    =%f\n', C_Sachant_M.T(2));

clampled = sparsevec(J, 2, 5);
C_Sachant_J = tabularFactorCondition(joint, C, clampled);
fprintf('C Sachant     J: p(C=1|J=1)    =%f\n\n', C_Sachant_J.T(2));

fprintf('(D) - Marginales Inconditionnelles\n');
clampled = zeros(5, 1);
p_C = tabularFactorCondition(joint, C, clampled);
fprintf('C: p(C=1)=%f\n', p_C.T(2));

clampled = zeros(5, 1);
p_T = tabularFactorCondition(joint, T, clampled);
fprintf('T: p(T=1)=%f\n', p_T.T(2));

clampled = zeros(5, 1);
p_A = tabularFactorCondition(joint, A, clampled);
fprintf('A: p(A=1)=%f\n', p_A.T(2));

clampled = zeros(5, 1);
p_M = tabularFactorCondition(joint, M, clampled);
fprintf('M: p(M=1)=%f\n', p_M.T(2));

clampled = zeros(5, 1);
p_J = tabularFactorCondition(joint, J, clampled);
fprintf('J: p(J=1)=%f\n', p_J.T(2));
