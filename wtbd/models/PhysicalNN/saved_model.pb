’
Õ¹
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
>
Maximum
x"T
y"T
z"T"
Ttype:
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
:
Minimum
x"T
y"T
z"T"
Ttype:

2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
³
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.5.02unknown8ß²
r
shift_sns/snsVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_nameshift_sns/sns
k
!shift_sns/sns/Read/ReadVariableOpReadVariableOpshift_sns/sns*
_output_shapes
:`*
dtype0

weighted_sum/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*%
shared_nameweighted_sum/weights
y
(weighted_sum/weights/Read/ReadVariableOpReadVariableOpweighted_sum/weights*
_output_shapes
:`*
dtype0
t
s_function/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_names_function/alpha
m
$s_function/alpha/Read/ReadVariableOpReadVariableOps_function/alpha*
_output_shapes
: *
dtype0
r
s_function/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_names_function/beta
k
#s_function/beta/Read/ReadVariableOpReadVariableOps_function/beta*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
t
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nametrue_positives
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:*
dtype0
v
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_negatives
o
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
:*
dtype0
x
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
v
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_positives
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
:*
dtype0

NoOpNoOp
«2
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ę1
valueÜ1BŁ1 BŅ1

layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer-5
layer-6
	optimizer
#	_self_saveable_object_factories

	variables
regularization_losses
trainable_variables
	keras_api

signatures
%
#_self_saveable_object_factories
w
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
w
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
%
#_self_saveable_object_factories
Ē
layer-0
layer-1
layer-2
layer-3
layer-4
 layer_with_weights-0
 layer-5
!layer-6
"layer-7
#layer_with_weights-1
#layer-8
$layer_with_weights-2
$layer-9
#%_self_saveable_object_factories
&	variables
'regularization_losses
(trainable_variables
)	keras_api
w
#*_self_saveable_object_factories
+	variables
,regularization_losses
-trainable_variables
.	keras_api
w
#/_self_saveable_object_factories
0	variables
1regularization_losses
2trainable_variables
3	keras_api
 
 

40
51
62
73
 

40
51
62
73
­

	variables
regularization_losses
8metrics
trainable_variables
9layer_metrics

:layers
;non_trainable_variables
<layer_regularization_losses
 
 
 
 
 
 
­
	variables
regularization_losses
=metrics
trainable_variables
>layer_metrics

?layers
@non_trainable_variables
Alayer_regularization_losses
 
 
 
 
­
	variables
regularization_losses
Bmetrics
trainable_variables
Clayer_metrics

Dlayers
Enon_trainable_variables
Flayer_regularization_losses
 
%
#G_self_saveable_object_factories
%
#H_self_saveable_object_factories
%
#I_self_saveable_object_factories
%
#J_self_saveable_object_factories

Kremove_axes
#L_self_saveable_object_factories
M	variables
Nregularization_losses
Otrainable_variables
P	keras_api
 
Q	sns_shape
4sns
Rexpand_axes
#S_self_saveable_object_factories
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
w
#X_self_saveable_object_factories
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
w
#]_self_saveable_object_factories
^	variables
_regularization_losses
`trainable_variables
a	keras_api

5w
baxes
#c_self_saveable_object_factories
d	variables
eregularization_losses
ftrainable_variables
g	keras_api

	6alpha
7beta
#h_self_saveable_object_factories
i	variables
jregularization_losses
ktrainable_variables
l	keras_api
 

40
51
62
73
 

40
51
62
73
­
&	variables
'regularization_losses
mmetrics
(trainable_variables
nlayer_metrics

olayers
pnon_trainable_variables
qlayer_regularization_losses
 
 
 
 
­
+	variables
,regularization_losses
rmetrics
-trainable_variables
slayer_metrics

tlayers
unon_trainable_variables
vlayer_regularization_losses
 
 
 
 
­
0	variables
1regularization_losses
wmetrics
2trainable_variables
xlayer_metrics

ylayers
znon_trainable_variables
{layer_regularization_losses
IG
VARIABLE_VALUEshift_sns/sns&variables/0/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEweighted_sum/weights&variables/1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEs_function/alpha&variables/2/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEs_function/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE

|0
}1
~2
3
 
1
0
1
2
3
4
5
6
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
²
M	variables
Nregularization_losses
metrics
Otrainable_variables
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
 
 
 

40
 

40
²
T	variables
Uregularization_losses
metrics
Vtrainable_variables
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
 
 
 
 
²
Y	variables
Zregularization_losses
metrics
[trainable_variables
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
 
 
 
 
²
^	variables
_regularization_losses
metrics
`trainable_variables
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
 
 

50
 

50
²
d	variables
eregularization_losses
metrics
ftrainable_variables
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
 

60
71
 

60
71
²
i	variables
jregularization_losses
metrics
ktrainable_variables
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
 
 
F
0
1
2
3
4
 5
!6
"7
#8
$9
 
 
 
 
 
 
 
 
 
 
 
 
8

total

count
 	variables
”	keras_api
I

¢total

£count
¤
_fn_kwargs
„	variables
¦	keras_api
\
§
thresholds
Øtrue_positives
©false_negatives
Ŗ	variables
«	keras_api
\
¬
thresholds
­true_positives
®false_positives
Æ	variables
°	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

 	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

¢0
£1

„	variables
 
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

Ø0
©1

Ŗ	variables
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUE

­0
®1

Æ	variables
s
serving_default_input_rsPlaceholder*#
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’

serving_default_input_spPlaceholder*+
_output_shapes
:’’’’’’’’’``*
dtype0* 
shape:’’’’’’’’’``

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_rsserving_default_input_spshift_sns/snsweighted_sum/weightss_function/betas_function/alpha*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_41810
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Å
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!shift_sns/sns/Read/ReadVariableOp(weighted_sum/weights/Read/ReadVariableOp$s_function/alpha/Read/ReadVariableOp#s_function/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp#false_positives/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_42130
Š
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameshift_sns/snsweighted_sum/weightss_function/alphas_function/betatotalcounttotal_1count_1true_positivesfalse_negativestrue_positives_1false_positives*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_42176ÕÜ

Ń
o
C__inference_subtract_layer_call_and_return_conditional_losses_42070
inputs_0
inputs_1
identityW
subSubinputs_0inputs_1*
T0*'
_output_shapes
:’’’’’’’’’`2
sub[
IdentityIdentitysub:z:0*
T0*'
_output_shapes
:’’’’’’’’’`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’`:’’’’’’’’’`:Q M
'
_output_shapes
:’’’’’’’’’`
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:’’’’’’’’’`
"
_user_specified_name
inputs/1
µ

E__inference_physicalnn_layer_call_and_return_conditional_losses_41669

inputs
inputs_1 
classifier3b_41653:` 
classifier3b_41655:`
classifier3b_41657: 
classifier3b_41659: 
identity¢$classifier3b/StatefulPartitionedCallŁ
#insert_channel_axis/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’``* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412682%
#insert_channel_axis/PartitionedCall
split/PartitionedCallPartitionedCall,insert_channel_axis/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’` * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412782
split/PartitionedCallÕ
$classifier3b/StatefulPartitionedCallStatefulPartitionedCallsplit/PartitionedCall:output:0split/PartitionedCall:output:1split/PartitionedCall:output:2inputs_1classifier3b_41653classifier3b_41655classifier3b_41657classifier3b_41659*
Tin

2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_classifier3b_layer_call_and_return_conditional_losses_414292&
$classifier3b/StatefulPartitionedCallņ
"classifier3b_dummy/PartitionedCallPartitionedCall-classifier3b/StatefulPartitionedCall:output:0-classifier3b/StatefulPartitionedCall:output:1-classifier3b/StatefulPartitionedCall:output:2*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413592$
"classifier3b_dummy/PartitionedCallø
any_of_3/PartitionedCallPartitionedCall+classifier3b_dummy/PartitionedCall:output:0+classifier3b_dummy/PartitionedCall:output:1+classifier3b_dummy/PartitionedCall:output:2*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413692
any_of_3/PartitionedCall
IdentityIdentity!any_of_3/PartitionedCall:output:0%^classifier3b/StatefulPartitionedCall*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:’’’’’’’’’``:’’’’’’’’’: : : : 2L
$classifier3b/StatefulPartitionedCall$classifier3b/StatefulPartitionedCall:S O
+
_output_shapes
:’’’’’’’’’``
 
_user_specified_nameinputs:KG
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
É
m
C__inference_subtract_layer_call_and_return_conditional_losses_41401

inputs
inputs_1
identityU
subSubinputsinputs_1*
T0*'
_output_shapes
:’’’’’’’’’`2
sub[
IdentityIdentitysub:z:0*
T0*'
_output_shapes
:’’’’’’’’’`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’`:’’’’’’’’’`:O K
'
_output_shapes
:’’’’’’’’’`
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’`
 
_user_specified_nameinputs
é
a
%__inference_split_layer_call_fn_39572

inputs
identity

identity_1

identity_2
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’` * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_split_layer_call_and_return_conditional_losses_395632
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:’’’’’’’’’` 2

Identityx

Identity_1IdentityPartitionedCall:output:1*
T0*/
_output_shapes
:’’’’’’’’’` 2

Identity_1x

Identity_2IdentityPartitionedCall:output:2*
T0*/
_output_shapes
:’’’’’’’’’` 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’``:W S
/
_output_shapes
:’’’’’’’’’``
 
_user_specified_nameinputs
ą
|
@__inference_split_layer_call_and_return_conditional_losses_39563

inputs
identity

identity_1

identity_2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ž’’’’’’’’2
split/split_dimŗ
splitSplitsplit/split_dim:output:0inputs*
T0*e
_output_shapesS
Q:’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’` *
	num_split2
splitj
IdentityIdentitysplit:output:0*
T0*/
_output_shapes
:’’’’’’’’’` 2

Identityn

Identity_1Identitysplit:output:1*
T0*/
_output_shapes
:’’’’’’’’’` 2

Identity_1n

Identity_2Identitysplit:output:2*
T0*/
_output_shapes
:’’’’’’’’’` 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’``:W S
/
_output_shapes
:’’’’’’’’’``
 
_user_specified_nameinputs
Į
D
(__inference_restored_function_body_41268

inputs
identityø
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:’’’’’’’’’``* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_insert_channel_axis_layer_call_and_return_conditional_losses_392192
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:’’’’’’’’’``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’``:S O
+
_output_shapes
:’’’’’’’’’``
 
_user_specified_nameinputs
³

,__inference_classifier3b_layer_call_fn_41952
inputs_0
inputs_1
inputs_2
inputs_3
unknown:`
	unknown_0:`
	unknown_1: 
	unknown_2: 
identity

identity_1

identity_2¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3unknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_classifier3b_layer_call_and_return_conditional_losses_414292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:’’’’’’’’’2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:’’’’’’’’’` 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:’’’’’’’’’` 
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:’’’’’’’’’` 
"
_user_specified_name
inputs/2:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/3
ÄP

G__inference_classifier3b_layer_call_and_return_conditional_losses_41429

inputs
inputs_1
inputs_2
inputs_3
shift_sns_41386:` 
weighted_sum_41408:`
s_function_41415: 
s_function_41417: 
identity

identity_1

identity_2¢"s_function/StatefulPartitionedCall¢$s_function/StatefulPartitionedCall_1¢$s_function/StatefulPartitionedCall_2¢!shift_sns/StatefulPartitionedCall¢#shift_sns/StatefulPartitionedCall_1¢#shift_sns/StatefulPartitionedCall_2¢$weighted_sum/StatefulPartitionedCall¢&weighted_sum/StatefulPartitionedCall_1¢&weighted_sum/StatefulPartitionedCall_2½
spectrum/PartitionedCallPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412862
spectrum/PartitionedCallģ
!shift_sns/StatefulPartitionedCallStatefulPartitionedCallinputs_3shift_sns_41386*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412942#
!shift_sns/StatefulPartitionedCallĮ
spectrum/PartitionedCall_1PartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412862
spectrum/PartitionedCall_1š
#shift_sns/StatefulPartitionedCall_1StatefulPartitionedCallinputs_3shift_sns_41386*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412942%
#shift_sns/StatefulPartitionedCall_1æ
spectrum/PartitionedCall_2PartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412862
spectrum/PartitionedCall_2š
#shift_sns/StatefulPartitionedCall_2StatefulPartitionedCallinputs_3shift_sns_41386*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412942%
#shift_sns/StatefulPartitionedCall_2
subtract/PartitionedCallPartitionedCall!spectrum/PartitionedCall:output:0*shift_sns/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_subtract_layer_call_and_return_conditional_losses_414012
subtract/PartitionedCall¦
subtract/PartitionedCall_1PartitionedCall#spectrum/PartitionedCall_1:output:0,shift_sns/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_subtract_layer_call_and_return_conditional_losses_414012
subtract/PartitionedCall_1¦
subtract/PartitionedCall_2PartitionedCall#spectrum/PartitionedCall_2:output:0,shift_sns/StatefulPartitionedCall_2:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_subtract_layer_call_and_return_conditional_losses_414012
subtract/PartitionedCall_2ą
relu_residual/PartitionedCallPartitionedCall!subtract/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413112
relu_residual/PartitionedCallę
relu_residual/PartitionedCall_1PartitionedCall#subtract/PartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413112!
relu_residual/PartitionedCall_1ę
relu_residual/PartitionedCall_2PartitionedCall#subtract/PartitionedCall_2:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413112!
relu_residual/PartitionedCall_2
$weighted_sum/StatefulPartitionedCallStatefulPartitionedCall&relu_residual/PartitionedCall:output:0weighted_sum_41408*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413212&
$weighted_sum/StatefulPartitionedCall
&weighted_sum/StatefulPartitionedCall_1StatefulPartitionedCall(relu_residual/PartitionedCall_1:output:0weighted_sum_41408*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413212(
&weighted_sum/StatefulPartitionedCall_1
&weighted_sum/StatefulPartitionedCall_2StatefulPartitionedCall(relu_residual/PartitionedCall_2:output:0weighted_sum_41408*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413212(
&weighted_sum/StatefulPartitionedCall_2¤
"s_function/StatefulPartitionedCallStatefulPartitionedCall-weighted_sum/StatefulPartitionedCall:output:0s_function_41415s_function_41417*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413372$
"s_function/StatefulPartitionedCallŖ
$s_function/StatefulPartitionedCall_1StatefulPartitionedCall/weighted_sum/StatefulPartitionedCall_1:output:0s_function_41415s_function_41417*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413372&
$s_function/StatefulPartitionedCall_1Ŗ
$s_function/StatefulPartitionedCall_2StatefulPartitionedCall/weighted_sum/StatefulPartitionedCall_2:output:0s_function_41415s_function_41417*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413372&
$s_function/StatefulPartitionedCall_2Ł
IdentityIdentity-s_function/StatefulPartitionedCall_2:output:0#^s_function/StatefulPartitionedCall%^s_function/StatefulPartitionedCall_1%^s_function/StatefulPartitionedCall_2"^shift_sns/StatefulPartitionedCall$^shift_sns/StatefulPartitionedCall_1$^shift_sns/StatefulPartitionedCall_2%^weighted_sum/StatefulPartitionedCall'^weighted_sum/StatefulPartitionedCall_1'^weighted_sum/StatefulPartitionedCall_2*
T0*#
_output_shapes
:’’’’’’’’’2

IdentityŻ

Identity_1Identity-s_function/StatefulPartitionedCall_1:output:0#^s_function/StatefulPartitionedCall%^s_function/StatefulPartitionedCall_1%^s_function/StatefulPartitionedCall_2"^shift_sns/StatefulPartitionedCall$^shift_sns/StatefulPartitionedCall_1$^shift_sns/StatefulPartitionedCall_2%^weighted_sum/StatefulPartitionedCall'^weighted_sum/StatefulPartitionedCall_1'^weighted_sum/StatefulPartitionedCall_2*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_1Ū

Identity_2Identity+s_function/StatefulPartitionedCall:output:0#^s_function/StatefulPartitionedCall%^s_function/StatefulPartitionedCall_1%^s_function/StatefulPartitionedCall_2"^shift_sns/StatefulPartitionedCall$^shift_sns/StatefulPartitionedCall_1$^shift_sns/StatefulPartitionedCall_2%^weighted_sum/StatefulPartitionedCall'^weighted_sum/StatefulPartitionedCall_1'^weighted_sum/StatefulPartitionedCall_2*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’: : : : 2H
"s_function/StatefulPartitionedCall"s_function/StatefulPartitionedCall2L
$s_function/StatefulPartitionedCall_1$s_function/StatefulPartitionedCall_12L
$s_function/StatefulPartitionedCall_2$s_function/StatefulPartitionedCall_22F
!shift_sns/StatefulPartitionedCall!shift_sns/StatefulPartitionedCall2J
#shift_sns/StatefulPartitionedCall_1#shift_sns/StatefulPartitionedCall_12J
#shift_sns/StatefulPartitionedCall_2#shift_sns/StatefulPartitionedCall_22L
$weighted_sum/StatefulPartitionedCall$weighted_sum/StatefulPartitionedCall2P
&weighted_sum/StatefulPartitionedCall_1&weighted_sum/StatefulPartitionedCall_12P
&weighted_sum/StatefulPartitionedCall_2&weighted_sum/StatefulPartitionedCall_2:W S
/
_output_shapes
:’’’’’’’’’` 
 
_user_specified_nameinputs:WS
/
_output_shapes
:’’’’’’’’’` 
 
_user_specified_nameinputs:WS
/
_output_shapes
:’’’’’’’’’` 
 
_user_specified_nameinputs:KG
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ÆJ

G__inference_classifier3b_layer_call_and_return_conditional_losses_42015
inputs_0
inputs_1
inputs_2
inputs_3
shift_sns_41979:` 
weighted_sum_41994:`
s_function_42001: 
s_function_42003: 
identity

identity_1

identity_2¢"s_function/StatefulPartitionedCall¢$s_function/StatefulPartitionedCall_1¢$s_function/StatefulPartitionedCall_2¢!shift_sns/StatefulPartitionedCall¢#shift_sns/StatefulPartitionedCall_1¢#shift_sns/StatefulPartitionedCall_2¢$weighted_sum/StatefulPartitionedCall¢&weighted_sum/StatefulPartitionedCall_1¢&weighted_sum/StatefulPartitionedCall_2½
spectrum/PartitionedCallPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412862
spectrum/PartitionedCallģ
!shift_sns/StatefulPartitionedCallStatefulPartitionedCallinputs_3shift_sns_41979*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412942#
!shift_sns/StatefulPartitionedCallĮ
spectrum/PartitionedCall_1PartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412862
spectrum/PartitionedCall_1š
#shift_sns/StatefulPartitionedCall_1StatefulPartitionedCallinputs_3shift_sns_41979*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412942%
#shift_sns/StatefulPartitionedCall_1Į
spectrum/PartitionedCall_2PartitionedCallinputs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412862
spectrum/PartitionedCall_2š
#shift_sns/StatefulPartitionedCall_2StatefulPartitionedCallinputs_3shift_sns_41979*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412942%
#shift_sns/StatefulPartitionedCall_2¤
subtract/subSub!spectrum/PartitionedCall:output:0*shift_sns/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2
subtract/sub¬
subtract/sub_1Sub#spectrum/PartitionedCall_1:output:0,shift_sns/StatefulPartitionedCall_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2
subtract/sub_1¬
subtract/sub_2Sub#spectrum/PartitionedCall_2:output:0,shift_sns/StatefulPartitionedCall_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2
subtract/sub_2Ļ
relu_residual/PartitionedCallPartitionedCallsubtract/sub:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413112
relu_residual/PartitionedCallÕ
relu_residual/PartitionedCall_1PartitionedCallsubtract/sub_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413112!
relu_residual/PartitionedCall_1Õ
relu_residual/PartitionedCall_2PartitionedCallsubtract/sub_2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413112!
relu_residual/PartitionedCall_2
$weighted_sum/StatefulPartitionedCallStatefulPartitionedCall&relu_residual/PartitionedCall:output:0weighted_sum_41994*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413212&
$weighted_sum/StatefulPartitionedCall
&weighted_sum/StatefulPartitionedCall_1StatefulPartitionedCall(relu_residual/PartitionedCall_1:output:0weighted_sum_41994*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413212(
&weighted_sum/StatefulPartitionedCall_1
&weighted_sum/StatefulPartitionedCall_2StatefulPartitionedCall(relu_residual/PartitionedCall_2:output:0weighted_sum_41994*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413212(
&weighted_sum/StatefulPartitionedCall_2¤
"s_function/StatefulPartitionedCallStatefulPartitionedCall-weighted_sum/StatefulPartitionedCall:output:0s_function_42001s_function_42003*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413372$
"s_function/StatefulPartitionedCallŖ
$s_function/StatefulPartitionedCall_1StatefulPartitionedCall/weighted_sum/StatefulPartitionedCall_1:output:0s_function_42001s_function_42003*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413372&
$s_function/StatefulPartitionedCall_1Ŗ
$s_function/StatefulPartitionedCall_2StatefulPartitionedCall/weighted_sum/StatefulPartitionedCall_2:output:0s_function_42001s_function_42003*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413372&
$s_function/StatefulPartitionedCall_2Ł
IdentityIdentity-s_function/StatefulPartitionedCall_2:output:0#^s_function/StatefulPartitionedCall%^s_function/StatefulPartitionedCall_1%^s_function/StatefulPartitionedCall_2"^shift_sns/StatefulPartitionedCall$^shift_sns/StatefulPartitionedCall_1$^shift_sns/StatefulPartitionedCall_2%^weighted_sum/StatefulPartitionedCall'^weighted_sum/StatefulPartitionedCall_1'^weighted_sum/StatefulPartitionedCall_2*
T0*#
_output_shapes
:’’’’’’’’’2

IdentityŻ

Identity_1Identity-s_function/StatefulPartitionedCall_1:output:0#^s_function/StatefulPartitionedCall%^s_function/StatefulPartitionedCall_1%^s_function/StatefulPartitionedCall_2"^shift_sns/StatefulPartitionedCall$^shift_sns/StatefulPartitionedCall_1$^shift_sns/StatefulPartitionedCall_2%^weighted_sum/StatefulPartitionedCall'^weighted_sum/StatefulPartitionedCall_1'^weighted_sum/StatefulPartitionedCall_2*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_1Ū

Identity_2Identity+s_function/StatefulPartitionedCall:output:0#^s_function/StatefulPartitionedCall%^s_function/StatefulPartitionedCall_1%^s_function/StatefulPartitionedCall_2"^shift_sns/StatefulPartitionedCall$^shift_sns/StatefulPartitionedCall_1$^shift_sns/StatefulPartitionedCall_2%^weighted_sum/StatefulPartitionedCall'^weighted_sum/StatefulPartitionedCall_1'^weighted_sum/StatefulPartitionedCall_2*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’: : : : 2H
"s_function/StatefulPartitionedCall"s_function/StatefulPartitionedCall2L
$s_function/StatefulPartitionedCall_1$s_function/StatefulPartitionedCall_12L
$s_function/StatefulPartitionedCall_2$s_function/StatefulPartitionedCall_22F
!shift_sns/StatefulPartitionedCall!shift_sns/StatefulPartitionedCall2J
#shift_sns/StatefulPartitionedCall_1#shift_sns/StatefulPartitionedCall_12J
#shift_sns/StatefulPartitionedCall_2#shift_sns/StatefulPartitionedCall_22L
$weighted_sum/StatefulPartitionedCall$weighted_sum/StatefulPartitionedCall2P
&weighted_sum/StatefulPartitionedCall_1&weighted_sum/StatefulPartitionedCall_12P
&weighted_sum/StatefulPartitionedCall_2&weighted_sum/StatefulPartitionedCall_2:Y U
/
_output_shapes
:’’’’’’’’’` 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:’’’’’’’’’` 
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:’’’’’’’’’` 
"
_user_specified_name
inputs/2:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/3
®
D
(__inference_restored_function_body_41286

inputs
identity„
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_spectrum_layer_call_and_return_conditional_losses_383172
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’` :W S
/
_output_shapes
:’’’’’’’’’` 
 
_user_specified_nameinputs
ą
d
H__inference_relu_residual_layer_call_and_return_conditional_losses_38350

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:’’’’’’’’’`2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’`:O K
'
_output_shapes
:’’’’’’’’’`
 
_user_specified_nameinputs

Ŗ
,__inference_classifier3b_layer_call_fn_41444
classifier3b_bm1
classifier3b_bm2
classifier3b_bm3
classifier3b_rs
unknown:`
	unknown_0:`
	unknown_1: 
	unknown_2: 
identity

identity_1

identity_2¢StatefulPartitionedCallņ
StatefulPartitionedCallStatefulPartitionedCallclassifier3b_bm1classifier3b_bm2classifier3b_bm3classifier3b_rsunknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_classifier3b_layer_call_and_return_conditional_losses_414292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:’’’’’’’’’2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:’’’’’’’’’` 
*
_user_specified_nameclassifier3b/bm1:a]
/
_output_shapes
:’’’’’’’’’` 
*
_user_specified_nameclassifier3b/bm2:a]
/
_output_shapes
:’’’’’’’’’` 
*
_user_specified_nameclassifier3b/bm3:TP
#
_output_shapes
:’’’’’’’’’
)
_user_specified_nameclassifier3b/rs

j
N__inference_insert_channel_axis_layer_call_and_return_conditional_losses_39219

inputs
identityk
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*/
_output_shapes
:’’’’’’’’’``2

ExpandDimso
IdentityIdentityExpandDims:output:0*
T0*/
_output_shapes
:’’’’’’’’’``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’``:S O
+
_output_shapes
:’’’’’’’’’``
 
_user_specified_nameinputs
£
D
(__inference_restored_function_body_41311

inputs
identityŖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_relu_residual_layer_call_and_return_conditional_losses_390612
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’`:O K
'
_output_shapes
:’’’’’’’’’`
 
_user_specified_nameinputs
½

E__inference_physicalnn_layer_call_and_return_conditional_losses_41794
input_sp
input_rs 
classifier3b_41778:` 
classifier3b_41780:`
classifier3b_41782: 
classifier3b_41784: 
identity¢$classifier3b/StatefulPartitionedCallŪ
#insert_channel_axis/PartitionedCallPartitionedCallinput_sp*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’``* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412682%
#insert_channel_axis/PartitionedCall
split/PartitionedCallPartitionedCall,insert_channel_axis/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’` * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412782
split/PartitionedCallÕ
$classifier3b/StatefulPartitionedCallStatefulPartitionedCallsplit/PartitionedCall:output:0split/PartitionedCall:output:1split/PartitionedCall:output:2input_rsclassifier3b_41778classifier3b_41780classifier3b_41782classifier3b_41784*
Tin

2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_classifier3b_layer_call_and_return_conditional_losses_415192&
$classifier3b/StatefulPartitionedCallņ
"classifier3b_dummy/PartitionedCallPartitionedCall-classifier3b/StatefulPartitionedCall:output:0-classifier3b/StatefulPartitionedCall:output:1-classifier3b/StatefulPartitionedCall:output:2*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413592$
"classifier3b_dummy/PartitionedCallø
any_of_3/PartitionedCallPartitionedCall+classifier3b_dummy/PartitionedCall:output:0+classifier3b_dummy/PartitionedCall:output:1+classifier3b_dummy/PartitionedCall:output:2*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413692
any_of_3/PartitionedCall
IdentityIdentity!any_of_3/PartitionedCall:output:0%^classifier3b/StatefulPartitionedCall*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:’’’’’’’’’``:’’’’’’’’’: : : : 2L
$classifier3b/StatefulPartitionedCall$classifier3b/StatefulPartitionedCall:U Q
+
_output_shapes
:’’’’’’’’’``
"
_user_specified_name
input_sp:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
input_rs
½

E__inference_physicalnn_layer_call_and_return_conditional_losses_41770
input_sp
input_rs 
classifier3b_41754:` 
classifier3b_41756:`
classifier3b_41758: 
classifier3b_41760: 
identity¢$classifier3b/StatefulPartitionedCallŪ
#insert_channel_axis/PartitionedCallPartitionedCallinput_sp*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’``* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412682%
#insert_channel_axis/PartitionedCall
split/PartitionedCallPartitionedCall,insert_channel_axis/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’` * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412782
split/PartitionedCallÕ
$classifier3b/StatefulPartitionedCallStatefulPartitionedCallsplit/PartitionedCall:output:0split/PartitionedCall:output:1split/PartitionedCall:output:2input_rsclassifier3b_41754classifier3b_41756classifier3b_41758classifier3b_41760*
Tin

2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_classifier3b_layer_call_and_return_conditional_losses_414292&
$classifier3b/StatefulPartitionedCallņ
"classifier3b_dummy/PartitionedCallPartitionedCall-classifier3b/StatefulPartitionedCall:output:0-classifier3b/StatefulPartitionedCall:output:1-classifier3b/StatefulPartitionedCall:output:2*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413592$
"classifier3b_dummy/PartitionedCallø
any_of_3/PartitionedCallPartitionedCall+classifier3b_dummy/PartitionedCall:output:0+classifier3b_dummy/PartitionedCall:output:1+classifier3b_dummy/PartitionedCall:output:2*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413692
any_of_3/PartitionedCall
IdentityIdentity!any_of_3/PartitionedCall:output:0%^classifier3b/StatefulPartitionedCall*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:’’’’’’’’’``:’’’’’’’’’: : : : 2L
$classifier3b/StatefulPartitionedCall$classifier3b/StatefulPartitionedCall:U Q
+
_output_shapes
:’’’’’’’’’``
"
_user_specified_name
input_sp:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
input_rs


G__inference_weighted_sum_layer_call_and_return_conditional_losses_39075

inputs
readvariableop_resource
identity¢ReadVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype02
ReadVariableOpZ
SoftmaxSoftmaxReadVariableOp:value:0*
T0*
_output_shapes
:`2	
Softmax{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2’
strided_sliceStridedSliceSoftmax:softmax:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:`*
ellipsis_mask*
new_axis_mask2
strided_slicec
mulMulstrided_slice:output:0inputs*
T0*'
_output_shapes
:’’’’’’’’’`2
mulx
Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2
Sum/reduction_indicesh
SumSummul:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:’’’’’’’’’2
Summ
IdentityIdentitySum:output:0^ReadVariableOp*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’`:2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’`
 
_user_specified_nameinputs
»Q
¦
G__inference_classifier3b_layer_call_and_return_conditional_losses_41597
classifier3b_bm1
classifier3b_bm2
classifier3b_bm3
classifier3b_rs
shift_sns_41561:` 
weighted_sum_41576:`
s_function_41583: 
s_function_41585: 
identity

identity_1

identity_2¢"s_function/StatefulPartitionedCall¢$s_function/StatefulPartitionedCall_1¢$s_function/StatefulPartitionedCall_2¢!shift_sns/StatefulPartitionedCall¢#shift_sns/StatefulPartitionedCall_1¢#shift_sns/StatefulPartitionedCall_2¢$weighted_sum/StatefulPartitionedCall¢&weighted_sum/StatefulPartitionedCall_1¢&weighted_sum/StatefulPartitionedCall_2Å
spectrum/PartitionedCallPartitionedCallclassifier3b_bm3*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412862
spectrum/PartitionedCalló
!shift_sns/StatefulPartitionedCallStatefulPartitionedCallclassifier3b_rsshift_sns_41561*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412942#
!shift_sns/StatefulPartitionedCallÉ
spectrum/PartitionedCall_1PartitionedCallclassifier3b_bm2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412862
spectrum/PartitionedCall_1÷
#shift_sns/StatefulPartitionedCall_1StatefulPartitionedCallclassifier3b_rsshift_sns_41561*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412942%
#shift_sns/StatefulPartitionedCall_1É
spectrum/PartitionedCall_2PartitionedCallclassifier3b_bm1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412862
spectrum/PartitionedCall_2÷
#shift_sns/StatefulPartitionedCall_2StatefulPartitionedCallclassifier3b_rsshift_sns_41561*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412942%
#shift_sns/StatefulPartitionedCall_2
subtract/PartitionedCallPartitionedCall!spectrum/PartitionedCall:output:0*shift_sns/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_subtract_layer_call_and_return_conditional_losses_414012
subtract/PartitionedCall¦
subtract/PartitionedCall_1PartitionedCall#spectrum/PartitionedCall_1:output:0,shift_sns/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_subtract_layer_call_and_return_conditional_losses_414012
subtract/PartitionedCall_1¦
subtract/PartitionedCall_2PartitionedCall#spectrum/PartitionedCall_2:output:0,shift_sns/StatefulPartitionedCall_2:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_subtract_layer_call_and_return_conditional_losses_414012
subtract/PartitionedCall_2ą
relu_residual/PartitionedCallPartitionedCall!subtract/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413112
relu_residual/PartitionedCallę
relu_residual/PartitionedCall_1PartitionedCall#subtract/PartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413112!
relu_residual/PartitionedCall_1ę
relu_residual/PartitionedCall_2PartitionedCall#subtract/PartitionedCall_2:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413112!
relu_residual/PartitionedCall_2
$weighted_sum/StatefulPartitionedCallStatefulPartitionedCall&relu_residual/PartitionedCall:output:0weighted_sum_41576*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413212&
$weighted_sum/StatefulPartitionedCall
&weighted_sum/StatefulPartitionedCall_1StatefulPartitionedCall(relu_residual/PartitionedCall_1:output:0weighted_sum_41576*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413212(
&weighted_sum/StatefulPartitionedCall_1
&weighted_sum/StatefulPartitionedCall_2StatefulPartitionedCall(relu_residual/PartitionedCall_2:output:0weighted_sum_41576*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413212(
&weighted_sum/StatefulPartitionedCall_2¤
"s_function/StatefulPartitionedCallStatefulPartitionedCall-weighted_sum/StatefulPartitionedCall:output:0s_function_41583s_function_41585*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413372$
"s_function/StatefulPartitionedCallŖ
$s_function/StatefulPartitionedCall_1StatefulPartitionedCall/weighted_sum/StatefulPartitionedCall_1:output:0s_function_41583s_function_41585*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413372&
$s_function/StatefulPartitionedCall_1Ŗ
$s_function/StatefulPartitionedCall_2StatefulPartitionedCall/weighted_sum/StatefulPartitionedCall_2:output:0s_function_41583s_function_41585*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413372&
$s_function/StatefulPartitionedCall_2Ł
IdentityIdentity-s_function/StatefulPartitionedCall_2:output:0#^s_function/StatefulPartitionedCall%^s_function/StatefulPartitionedCall_1%^s_function/StatefulPartitionedCall_2"^shift_sns/StatefulPartitionedCall$^shift_sns/StatefulPartitionedCall_1$^shift_sns/StatefulPartitionedCall_2%^weighted_sum/StatefulPartitionedCall'^weighted_sum/StatefulPartitionedCall_1'^weighted_sum/StatefulPartitionedCall_2*
T0*#
_output_shapes
:’’’’’’’’’2

IdentityŻ

Identity_1Identity-s_function/StatefulPartitionedCall_1:output:0#^s_function/StatefulPartitionedCall%^s_function/StatefulPartitionedCall_1%^s_function/StatefulPartitionedCall_2"^shift_sns/StatefulPartitionedCall$^shift_sns/StatefulPartitionedCall_1$^shift_sns/StatefulPartitionedCall_2%^weighted_sum/StatefulPartitionedCall'^weighted_sum/StatefulPartitionedCall_1'^weighted_sum/StatefulPartitionedCall_2*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_1Ū

Identity_2Identity+s_function/StatefulPartitionedCall:output:0#^s_function/StatefulPartitionedCall%^s_function/StatefulPartitionedCall_1%^s_function/StatefulPartitionedCall_2"^shift_sns/StatefulPartitionedCall$^shift_sns/StatefulPartitionedCall_1$^shift_sns/StatefulPartitionedCall_2%^weighted_sum/StatefulPartitionedCall'^weighted_sum/StatefulPartitionedCall_1'^weighted_sum/StatefulPartitionedCall_2*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’: : : : 2H
"s_function/StatefulPartitionedCall"s_function/StatefulPartitionedCall2L
$s_function/StatefulPartitionedCall_1$s_function/StatefulPartitionedCall_12L
$s_function/StatefulPartitionedCall_2$s_function/StatefulPartitionedCall_22F
!shift_sns/StatefulPartitionedCall!shift_sns/StatefulPartitionedCall2J
#shift_sns/StatefulPartitionedCall_1#shift_sns/StatefulPartitionedCall_12J
#shift_sns/StatefulPartitionedCall_2#shift_sns/StatefulPartitionedCall_22L
$weighted_sum/StatefulPartitionedCall$weighted_sum/StatefulPartitionedCall2P
&weighted_sum/StatefulPartitionedCall_1&weighted_sum/StatefulPartitionedCall_12P
&weighted_sum/StatefulPartitionedCall_2&weighted_sum/StatefulPartitionedCall_2:a ]
/
_output_shapes
:’’’’’’’’’` 
*
_user_specified_nameclassifier3b/bm1:a]
/
_output_shapes
:’’’’’’’’’` 
*
_user_specified_nameclassifier3b/bm2:a]
/
_output_shapes
:’’’’’’’’’` 
*
_user_specified_nameclassifier3b/bm3:TP
#
_output_shapes
:’’’’’’’’’
)
_user_specified_nameclassifier3b/rs
Ė
_
C__inference_spectrum_layer_call_and_return_conditional_losses_38640

inputs
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
	truediv/ys
truedivRealDivinputstruediv/y:output:0*
T0*/
_output_shapes
:’’’’’’’’’` 2	
truedivS
pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
pow/xh
powPowpow/x:output:0truediv:z:0*
T0*/
_output_shapes
:’’’’’’’’’` 2
pow{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ž’’’’’’’’2
Mean/reduction_indices
MeanMeanpow:z:0Mean/reduction_indices:output:0*
T0*/
_output_shapes
:’’’’’’’’’`*
	keep_dims(2
Mean
SqueezeSqueezeMean:output:0*
T0*'
_output_shapes
:’’’’’’’’’`*(
squeeze_dims
ž’’’’’’’’’’’’’’’’’2	
SqueezeU
LogLogSqueeze:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2
LogS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
ConstF
Log_1LogConst:output:0*
T0*
_output_shapes
: 2
Log_1g
	truediv_1RealDivLog:y:0	Log_1:y:0*
T0*'
_output_shapes
:’’’’’’’’’`2
	truediv_1S
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
mul/xb
mulMulmul/x:output:0truediv_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’`2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:’’’’’’’’’`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’` :W S
/
_output_shapes
:’’’’’’’’’` 
 
_user_specified_nameinputs
ė
y
)__inference_shift_sns_layer_call_fn_40217

inputs
unknown:`
identity¢StatefulPartitionedCallź
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_shift_sns_layer_call_and_return_conditional_losses_400542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’:22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ķ
D
(__inference_spectrum_layer_call_fn_38645

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_spectrum_layer_call_and_return_conditional_losses_386402
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’` :W S
/
_output_shapes
:’’’’’’’’’` 
 
_user_specified_nameinputs
Ē
I
-__inference_relu_residual_layer_call_fn_38355

inputs
identityÉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_relu_residual_layer_call_and_return_conditional_losses_383502
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’`:O K
'
_output_shapes
:’’’’’’’’’`
 
_user_specified_nameinputs
ė
O
3__inference_insert_channel_axis_layer_call_fn_40537

inputs
identity×
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’``* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_insert_channel_axis_layer_call_and_return_conditional_losses_405322
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:’’’’’’’’’``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’``:S O
+
_output_shapes
:’’’’’’’’’``
 
_user_specified_nameinputs
ń
|
,__inference_weighted_sum_layer_call_fn_39081

inputs
unknown:`
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_weighted_sum_layer_call_and_return_conditional_losses_390752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’`:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’`
 
_user_specified_nameinputs
ą
d
H__inference_relu_residual_layer_call_and_return_conditional_losses_39061

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:’’’’’’’’’`2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’`:O K
'
_output_shapes
:’’’’’’’’’`
 
_user_specified_nameinputs
Öb
Ü
 __inference__wrapped_model_41372
input_sp
input_rs5
'physicalnn_classifier3b_shift_sns_41295:`8
*physicalnn_classifier3b_weighted_sum_41322:`2
(physicalnn_classifier3b_s_function_41338: 2
(physicalnn_classifier3b_s_function_41340: 
identity¢:physicalnn/classifier3b/s_function/StatefulPartitionedCall¢<physicalnn/classifier3b/s_function/StatefulPartitionedCall_1¢<physicalnn/classifier3b/s_function/StatefulPartitionedCall_2¢9physicalnn/classifier3b/shift_sns/StatefulPartitionedCall¢;physicalnn/classifier3b/shift_sns/StatefulPartitionedCall_1¢;physicalnn/classifier3b/shift_sns/StatefulPartitionedCall_2¢<physicalnn/classifier3b/weighted_sum/StatefulPartitionedCall¢>physicalnn/classifier3b/weighted_sum/StatefulPartitionedCall_1¢>physicalnn/classifier3b/weighted_sum/StatefulPartitionedCall_2ń
.physicalnn/insert_channel_axis/PartitionedCallPartitionedCallinput_sp*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’``* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_4126820
.physicalnn/insert_channel_axis/PartitionedCall¼
 physicalnn/split/PartitionedCallPartitionedCall7physicalnn/insert_channel_axis/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’` * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412782"
 physicalnn/split/PartitionedCall
0physicalnn/classifier3b/spectrum/PartitionedCallPartitionedCall)physicalnn/split/PartitionedCall:output:2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_4128622
0physicalnn/classifier3b/spectrum/PartitionedCall“
9physicalnn/classifier3b/shift_sns/StatefulPartitionedCallStatefulPartitionedCallinput_rs'physicalnn_classifier3b_shift_sns_41295*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412942;
9physicalnn/classifier3b/shift_sns/StatefulPartitionedCall
2physicalnn/classifier3b/spectrum/PartitionedCall_1PartitionedCall)physicalnn/split/PartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_4128624
2physicalnn/classifier3b/spectrum/PartitionedCall_1ø
;physicalnn/classifier3b/shift_sns/StatefulPartitionedCall_1StatefulPartitionedCallinput_rs'physicalnn_classifier3b_shift_sns_41295*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412942=
;physicalnn/classifier3b/shift_sns/StatefulPartitionedCall_1
2physicalnn/classifier3b/spectrum/PartitionedCall_2PartitionedCall)physicalnn/split/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_4128624
2physicalnn/classifier3b/spectrum/PartitionedCall_2ø
;physicalnn/classifier3b/shift_sns/StatefulPartitionedCall_2StatefulPartitionedCallinput_rs'physicalnn_classifier3b_shift_sns_41295*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412942=
;physicalnn/classifier3b/shift_sns/StatefulPartitionedCall_2
$physicalnn/classifier3b/subtract/subSub9physicalnn/classifier3b/spectrum/PartitionedCall:output:0Bphysicalnn/classifier3b/shift_sns/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2&
$physicalnn/classifier3b/subtract/sub
&physicalnn/classifier3b/subtract/sub_1Sub;physicalnn/classifier3b/spectrum/PartitionedCall_1:output:0Dphysicalnn/classifier3b/shift_sns/StatefulPartitionedCall_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2(
&physicalnn/classifier3b/subtract/sub_1
&physicalnn/classifier3b/subtract/sub_2Sub;physicalnn/classifier3b/spectrum/PartitionedCall_2:output:0Dphysicalnn/classifier3b/shift_sns/StatefulPartitionedCall_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2(
&physicalnn/classifier3b/subtract/sub_2
5physicalnn/classifier3b/relu_residual/PartitionedCallPartitionedCall(physicalnn/classifier3b/subtract/sub:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_4131127
5physicalnn/classifier3b/relu_residual/PartitionedCall
7physicalnn/classifier3b/relu_residual/PartitionedCall_1PartitionedCall*physicalnn/classifier3b/subtract/sub_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_4131129
7physicalnn/classifier3b/relu_residual/PartitionedCall_1
7physicalnn/classifier3b/relu_residual/PartitionedCall_2PartitionedCall*physicalnn/classifier3b/subtract/sub_2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_4131129
7physicalnn/classifier3b/relu_residual/PartitionedCall_2ļ
<physicalnn/classifier3b/weighted_sum/StatefulPartitionedCallStatefulPartitionedCall>physicalnn/classifier3b/relu_residual/PartitionedCall:output:0*physicalnn_classifier3b_weighted_sum_41322*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413212>
<physicalnn/classifier3b/weighted_sum/StatefulPartitionedCallõ
>physicalnn/classifier3b/weighted_sum/StatefulPartitionedCall_1StatefulPartitionedCall@physicalnn/classifier3b/relu_residual/PartitionedCall_1:output:0*physicalnn_classifier3b_weighted_sum_41322*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413212@
>physicalnn/classifier3b/weighted_sum/StatefulPartitionedCall_1õ
>physicalnn/classifier3b/weighted_sum/StatefulPartitionedCall_2StatefulPartitionedCall@physicalnn/classifier3b/relu_residual/PartitionedCall_2:output:0*physicalnn_classifier3b_weighted_sum_41322*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413212@
>physicalnn/classifier3b/weighted_sum/StatefulPartitionedCall_2
:physicalnn/classifier3b/s_function/StatefulPartitionedCallStatefulPartitionedCallEphysicalnn/classifier3b/weighted_sum/StatefulPartitionedCall:output:0(physicalnn_classifier3b_s_function_41338(physicalnn_classifier3b_s_function_41340*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413372<
:physicalnn/classifier3b/s_function/StatefulPartitionedCall¢
<physicalnn/classifier3b/s_function/StatefulPartitionedCall_1StatefulPartitionedCallGphysicalnn/classifier3b/weighted_sum/StatefulPartitionedCall_1:output:0(physicalnn_classifier3b_s_function_41338(physicalnn_classifier3b_s_function_41340*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413372>
<physicalnn/classifier3b/s_function/StatefulPartitionedCall_1¢
<physicalnn/classifier3b/s_function/StatefulPartitionedCall_2StatefulPartitionedCallGphysicalnn/classifier3b/weighted_sum/StatefulPartitionedCall_2:output:0(physicalnn_classifier3b_s_function_41338(physicalnn_classifier3b_s_function_41340*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413372>
<physicalnn/classifier3b/s_function/StatefulPartitionedCall_2Ī
-physicalnn/classifier3b_dummy/PartitionedCallPartitionedCallEphysicalnn/classifier3b/s_function/StatefulPartitionedCall_2:output:0Ephysicalnn/classifier3b/s_function/StatefulPartitionedCall_1:output:0Cphysicalnn/classifier3b/s_function/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413592/
-physicalnn/classifier3b_dummy/PartitionedCallļ
#physicalnn/any_of_3/PartitionedCallPartitionedCall6physicalnn/classifier3b_dummy/PartitionedCall:output:06physicalnn/classifier3b_dummy/PartitionedCall:output:16physicalnn/classifier3b_dummy/PartitionedCall:output:2*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413692%
#physicalnn/any_of_3/PartitionedCall°
IdentityIdentity,physicalnn/any_of_3/PartitionedCall:output:0;^physicalnn/classifier3b/s_function/StatefulPartitionedCall=^physicalnn/classifier3b/s_function/StatefulPartitionedCall_1=^physicalnn/classifier3b/s_function/StatefulPartitionedCall_2:^physicalnn/classifier3b/shift_sns/StatefulPartitionedCall<^physicalnn/classifier3b/shift_sns/StatefulPartitionedCall_1<^physicalnn/classifier3b/shift_sns/StatefulPartitionedCall_2=^physicalnn/classifier3b/weighted_sum/StatefulPartitionedCall?^physicalnn/classifier3b/weighted_sum/StatefulPartitionedCall_1?^physicalnn/classifier3b/weighted_sum/StatefulPartitionedCall_2*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:’’’’’’’’’``:’’’’’’’’’: : : : 2x
:physicalnn/classifier3b/s_function/StatefulPartitionedCall:physicalnn/classifier3b/s_function/StatefulPartitionedCall2|
<physicalnn/classifier3b/s_function/StatefulPartitionedCall_1<physicalnn/classifier3b/s_function/StatefulPartitionedCall_12|
<physicalnn/classifier3b/s_function/StatefulPartitionedCall_2<physicalnn/classifier3b/s_function/StatefulPartitionedCall_22v
9physicalnn/classifier3b/shift_sns/StatefulPartitionedCall9physicalnn/classifier3b/shift_sns/StatefulPartitionedCall2z
;physicalnn/classifier3b/shift_sns/StatefulPartitionedCall_1;physicalnn/classifier3b/shift_sns/StatefulPartitionedCall_12z
;physicalnn/classifier3b/shift_sns/StatefulPartitionedCall_2;physicalnn/classifier3b/shift_sns/StatefulPartitionedCall_22|
<physicalnn/classifier3b/weighted_sum/StatefulPartitionedCall<physicalnn/classifier3b/weighted_sum/StatefulPartitionedCall2
>physicalnn/classifier3b/weighted_sum/StatefulPartitionedCall_1>physicalnn/classifier3b/weighted_sum/StatefulPartitionedCall_12
>physicalnn/classifier3b/weighted_sum/StatefulPartitionedCall_2>physicalnn/classifier3b/weighted_sum/StatefulPartitionedCall_2:U Q
+
_output_shapes
:’’’’’’’’’``
"
_user_specified_name
input_sp:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
input_rs
į
Ķ
*__inference_physicalnn_layer_call_fn_41746
input_sp
input_rs
unknown:`
	unknown_0:`
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_spinput_rsunknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_physicalnn_layer_call_and_return_conditional_losses_417212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:’’’’’’’’’``:’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:’’’’’’’’’``
"
_user_specified_name
input_sp:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
input_rs
³

,__inference_classifier3b_layer_call_fn_41972
inputs_0
inputs_1
inputs_2
inputs_3
unknown:`
	unknown_0:`
	unknown_1: 
	unknown_2: 
identity

identity_1

identity_2¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3unknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_classifier3b_layer_call_and_return_conditional_losses_415192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:’’’’’’’’’2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:’’’’’’’’’` 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:’’’’’’’’’` 
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:’’’’’’’’’` 
"
_user_specified_name
inputs/2:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/3
µ
Ę
#__inference_signature_wrapper_41810
input_rs
input_sp
unknown:`
	unknown_0:`
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinput_spinput_rsunknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_413722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:’’’’’’’’’:’’’’’’’’’``: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
input_rs:UQ
+
_output_shapes
:’’’’’’’’’``
"
_user_specified_name
input_sp
Ā
T
(__inference_subtract_layer_call_fn_42064
inputs_0
inputs_1
identityŃ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_subtract_layer_call_and_return_conditional_losses_414012
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’`:’’’’’’’’’`:Q M
'
_output_shapes
:’’’’’’’’’`
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:’’’’’’’’’`
"
_user_specified_name
inputs/1
é
Ō
E__inference_s_function_layer_call_and_return_conditional_losses_38736

inputs%
pow_readvariableop_resource: !
readvariableop_resource: 
identity¢ReadVariableOp¢pow/ReadVariableOp|
pow/ReadVariableOpReadVariableOppow_readvariableop_resource*
_output_shapes
: *
dtype02
pow/ReadVariableOpc
powPowinputspow/ReadVariableOp:value:0*
T0*#
_output_shapes
:’’’’’’’’’2
powp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp`
mulMulReadVariableOp:value:0pow:z:0*
T0*#
_output_shapes
:’’’’’’’’’2
mulK
TanhTanhmul:z:0*
T0*#
_output_shapes
:’’’’’’’’’2
Tanh~
IdentityIdentityTanh:y:0^ReadVariableOp^pow/ReadVariableOp*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’::2 
ReadVariableOpReadVariableOp2(
pow/ReadVariableOppow/ReadVariableOp:K G
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ž
§
M__inference_classifier3b_dummy_layer_call_and_return_conditional_losses_39586
inputs_0
inputs_1
inputs_2
identity

identity_1

identity_2X
IdentityIdentityinputs_0*
T0*#
_output_shapes
:’’’’’’’’’2

Identity\

Identity_1Identityinputs_1*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_1\

Identity_2Identityinputs_2*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:M I
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/1:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/2
·
}
C__inference_any_of_3_layer_call_and_return_conditional_losses_40766
inputs_0
inputs_1
inputs_2
identityU
addAddV2inputs_0inputs_1*
T0*#
_output_shapes
:’’’’’’’’’2
addX
add_1AddV2add:z:0inputs_2*
T0*#
_output_shapes
:’’’’’’’’’2
add_1S
mulMulinputs_0inputs_1*
T0*#
_output_shapes
:’’’’’’’’’2
mulW
mul_1Mulinputs_1inputs_2*
T0*#
_output_shapes
:’’’’’’’’’2
mul_1Y
add_2AddV2mul:z:0	mul_1:z:0*
T0*#
_output_shapes
:’’’’’’’’’2
add_2W
mul_2Mulinputs_2inputs_0*
T0*#
_output_shapes
:’’’’’’’’’2
mul_2[
add_3AddV2	add_2:z:0	mul_2:z:0*
T0*#
_output_shapes
:’’’’’’’’’2
add_3U
subSub	add_1:z:0	add_3:z:0*
T0*#
_output_shapes
:’’’’’’’’’2
subW
mul_3Mulinputs_0inputs_1*
T0*#
_output_shapes
:’’’’’’’’’2
mul_3X
mul_4Mul	mul_3:z:0inputs_2*
T0*#
_output_shapes
:’’’’’’’’’2
mul_4Y
add_4AddV2sub:z:0	mul_4:z:0*
T0*#
_output_shapes
:’’’’’’’’’2
add_4w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value/Minimum/y
clip_by_value/MinimumMinimum	add_4:z:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:’’’’’’’’’2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:’’’’’’’’’2
clip_by_valuea
IdentityIdentityclip_by_value:z:0*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:M I
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/1:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/2
ÄP

G__inference_classifier3b_layer_call_and_return_conditional_losses_41519

inputs
inputs_1
inputs_2
inputs_3
shift_sns_41483:` 
weighted_sum_41498:`
s_function_41505: 
s_function_41507: 
identity

identity_1

identity_2¢"s_function/StatefulPartitionedCall¢$s_function/StatefulPartitionedCall_1¢$s_function/StatefulPartitionedCall_2¢!shift_sns/StatefulPartitionedCall¢#shift_sns/StatefulPartitionedCall_1¢#shift_sns/StatefulPartitionedCall_2¢$weighted_sum/StatefulPartitionedCall¢&weighted_sum/StatefulPartitionedCall_1¢&weighted_sum/StatefulPartitionedCall_2½
spectrum/PartitionedCallPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412862
spectrum/PartitionedCallģ
!shift_sns/StatefulPartitionedCallStatefulPartitionedCallinputs_3shift_sns_41483*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412942#
!shift_sns/StatefulPartitionedCallĮ
spectrum/PartitionedCall_1PartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412862
spectrum/PartitionedCall_1š
#shift_sns/StatefulPartitionedCall_1StatefulPartitionedCallinputs_3shift_sns_41483*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412942%
#shift_sns/StatefulPartitionedCall_1æ
spectrum/PartitionedCall_2PartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412862
spectrum/PartitionedCall_2š
#shift_sns/StatefulPartitionedCall_2StatefulPartitionedCallinputs_3shift_sns_41483*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412942%
#shift_sns/StatefulPartitionedCall_2
subtract/PartitionedCallPartitionedCall!spectrum/PartitionedCall:output:0*shift_sns/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_subtract_layer_call_and_return_conditional_losses_414012
subtract/PartitionedCall¦
subtract/PartitionedCall_1PartitionedCall#spectrum/PartitionedCall_1:output:0,shift_sns/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_subtract_layer_call_and_return_conditional_losses_414012
subtract/PartitionedCall_1¦
subtract/PartitionedCall_2PartitionedCall#spectrum/PartitionedCall_2:output:0,shift_sns/StatefulPartitionedCall_2:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_subtract_layer_call_and_return_conditional_losses_414012
subtract/PartitionedCall_2ą
relu_residual/PartitionedCallPartitionedCall!subtract/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413112
relu_residual/PartitionedCallę
relu_residual/PartitionedCall_1PartitionedCall#subtract/PartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413112!
relu_residual/PartitionedCall_1ę
relu_residual/PartitionedCall_2PartitionedCall#subtract/PartitionedCall_2:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413112!
relu_residual/PartitionedCall_2
$weighted_sum/StatefulPartitionedCallStatefulPartitionedCall&relu_residual/PartitionedCall:output:0weighted_sum_41498*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413212&
$weighted_sum/StatefulPartitionedCall
&weighted_sum/StatefulPartitionedCall_1StatefulPartitionedCall(relu_residual/PartitionedCall_1:output:0weighted_sum_41498*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413212(
&weighted_sum/StatefulPartitionedCall_1
&weighted_sum/StatefulPartitionedCall_2StatefulPartitionedCall(relu_residual/PartitionedCall_2:output:0weighted_sum_41498*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413212(
&weighted_sum/StatefulPartitionedCall_2¤
"s_function/StatefulPartitionedCallStatefulPartitionedCall-weighted_sum/StatefulPartitionedCall:output:0s_function_41505s_function_41507*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413372$
"s_function/StatefulPartitionedCallŖ
$s_function/StatefulPartitionedCall_1StatefulPartitionedCall/weighted_sum/StatefulPartitionedCall_1:output:0s_function_41505s_function_41507*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413372&
$s_function/StatefulPartitionedCall_1Ŗ
$s_function/StatefulPartitionedCall_2StatefulPartitionedCall/weighted_sum/StatefulPartitionedCall_2:output:0s_function_41505s_function_41507*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413372&
$s_function/StatefulPartitionedCall_2Ł
IdentityIdentity-s_function/StatefulPartitionedCall_2:output:0#^s_function/StatefulPartitionedCall%^s_function/StatefulPartitionedCall_1%^s_function/StatefulPartitionedCall_2"^shift_sns/StatefulPartitionedCall$^shift_sns/StatefulPartitionedCall_1$^shift_sns/StatefulPartitionedCall_2%^weighted_sum/StatefulPartitionedCall'^weighted_sum/StatefulPartitionedCall_1'^weighted_sum/StatefulPartitionedCall_2*
T0*#
_output_shapes
:’’’’’’’’’2

IdentityŻ

Identity_1Identity-s_function/StatefulPartitionedCall_1:output:0#^s_function/StatefulPartitionedCall%^s_function/StatefulPartitionedCall_1%^s_function/StatefulPartitionedCall_2"^shift_sns/StatefulPartitionedCall$^shift_sns/StatefulPartitionedCall_1$^shift_sns/StatefulPartitionedCall_2%^weighted_sum/StatefulPartitionedCall'^weighted_sum/StatefulPartitionedCall_1'^weighted_sum/StatefulPartitionedCall_2*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_1Ū

Identity_2Identity+s_function/StatefulPartitionedCall:output:0#^s_function/StatefulPartitionedCall%^s_function/StatefulPartitionedCall_1%^s_function/StatefulPartitionedCall_2"^shift_sns/StatefulPartitionedCall$^shift_sns/StatefulPartitionedCall_1$^shift_sns/StatefulPartitionedCall_2%^weighted_sum/StatefulPartitionedCall'^weighted_sum/StatefulPartitionedCall_1'^weighted_sum/StatefulPartitionedCall_2*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’: : : : 2H
"s_function/StatefulPartitionedCall"s_function/StatefulPartitionedCall2L
$s_function/StatefulPartitionedCall_1$s_function/StatefulPartitionedCall_12L
$s_function/StatefulPartitionedCall_2$s_function/StatefulPartitionedCall_22F
!shift_sns/StatefulPartitionedCall!shift_sns/StatefulPartitionedCall2J
#shift_sns/StatefulPartitionedCall_1#shift_sns/StatefulPartitionedCall_12J
#shift_sns/StatefulPartitionedCall_2#shift_sns/StatefulPartitionedCall_22L
$weighted_sum/StatefulPartitionedCall$weighted_sum/StatefulPartitionedCall2P
&weighted_sum/StatefulPartitionedCall_1&weighted_sum/StatefulPartitionedCall_12P
&weighted_sum/StatefulPartitionedCall_2&weighted_sum/StatefulPartitionedCall_2:W S
/
_output_shapes
:’’’’’’’’’` 
 
_user_specified_nameinputs:WS
/
_output_shapes
:’’’’’’’’’` 
 
_user_specified_nameinputs:WS
/
_output_shapes
:’’’’’’’’’` 
 
_user_specified_nameinputs:KG
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ė
_
C__inference_spectrum_layer_call_and_return_conditional_losses_38317

inputs
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
	truediv/ys
truedivRealDivinputstruediv/y:output:0*
T0*/
_output_shapes
:’’’’’’’’’` 2	
truedivS
pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
pow/xh
powPowpow/x:output:0truediv:z:0*
T0*/
_output_shapes
:’’’’’’’’’` 2
pow{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ž’’’’’’’’2
Mean/reduction_indices
MeanMeanpow:z:0Mean/reduction_indices:output:0*
T0*/
_output_shapes
:’’’’’’’’’`*
	keep_dims(2
Mean
SqueezeSqueezeMean:output:0*
T0*'
_output_shapes
:’’’’’’’’’`*(
squeeze_dims
ž’’’’’’’’’’’’’’’’’2	
SqueezeU
LogLogSqueeze:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2
LogS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
ConstF
Log_1LogConst:output:0*
T0*
_output_shapes
: 2
Log_1g
	truediv_1RealDivLog:y:0	Log_1:y:0*
T0*'
_output_shapes
:’’’’’’’’’`2
	truediv_1S
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
mul/xb
mulMulmul/x:output:0truediv_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’`2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:’’’’’’’’’`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’` :W S
/
_output_shapes
:’’’’’’’’’` 
 
_user_specified_nameinputs
ÆJ

G__inference_classifier3b_layer_call_and_return_conditional_losses_42058
inputs_0
inputs_1
inputs_2
inputs_3
shift_sns_42022:` 
weighted_sum_42037:`
s_function_42044: 
s_function_42046: 
identity

identity_1

identity_2¢"s_function/StatefulPartitionedCall¢$s_function/StatefulPartitionedCall_1¢$s_function/StatefulPartitionedCall_2¢!shift_sns/StatefulPartitionedCall¢#shift_sns/StatefulPartitionedCall_1¢#shift_sns/StatefulPartitionedCall_2¢$weighted_sum/StatefulPartitionedCall¢&weighted_sum/StatefulPartitionedCall_1¢&weighted_sum/StatefulPartitionedCall_2½
spectrum/PartitionedCallPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412862
spectrum/PartitionedCallģ
!shift_sns/StatefulPartitionedCallStatefulPartitionedCallinputs_3shift_sns_42022*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412942#
!shift_sns/StatefulPartitionedCallĮ
spectrum/PartitionedCall_1PartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412862
spectrum/PartitionedCall_1š
#shift_sns/StatefulPartitionedCall_1StatefulPartitionedCallinputs_3shift_sns_42022*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412942%
#shift_sns/StatefulPartitionedCall_1Į
spectrum/PartitionedCall_2PartitionedCallinputs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412862
spectrum/PartitionedCall_2š
#shift_sns/StatefulPartitionedCall_2StatefulPartitionedCallinputs_3shift_sns_42022*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412942%
#shift_sns/StatefulPartitionedCall_2¤
subtract/subSub!spectrum/PartitionedCall:output:0*shift_sns/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2
subtract/sub¬
subtract/sub_1Sub#spectrum/PartitionedCall_1:output:0,shift_sns/StatefulPartitionedCall_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2
subtract/sub_1¬
subtract/sub_2Sub#spectrum/PartitionedCall_2:output:0,shift_sns/StatefulPartitionedCall_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2
subtract/sub_2Ļ
relu_residual/PartitionedCallPartitionedCallsubtract/sub:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413112
relu_residual/PartitionedCallÕ
relu_residual/PartitionedCall_1PartitionedCallsubtract/sub_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413112!
relu_residual/PartitionedCall_1Õ
relu_residual/PartitionedCall_2PartitionedCallsubtract/sub_2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413112!
relu_residual/PartitionedCall_2
$weighted_sum/StatefulPartitionedCallStatefulPartitionedCall&relu_residual/PartitionedCall:output:0weighted_sum_42037*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413212&
$weighted_sum/StatefulPartitionedCall
&weighted_sum/StatefulPartitionedCall_1StatefulPartitionedCall(relu_residual/PartitionedCall_1:output:0weighted_sum_42037*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413212(
&weighted_sum/StatefulPartitionedCall_1
&weighted_sum/StatefulPartitionedCall_2StatefulPartitionedCall(relu_residual/PartitionedCall_2:output:0weighted_sum_42037*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413212(
&weighted_sum/StatefulPartitionedCall_2¤
"s_function/StatefulPartitionedCallStatefulPartitionedCall-weighted_sum/StatefulPartitionedCall:output:0s_function_42044s_function_42046*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413372$
"s_function/StatefulPartitionedCallŖ
$s_function/StatefulPartitionedCall_1StatefulPartitionedCall/weighted_sum/StatefulPartitionedCall_1:output:0s_function_42044s_function_42046*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413372&
$s_function/StatefulPartitionedCall_1Ŗ
$s_function/StatefulPartitionedCall_2StatefulPartitionedCall/weighted_sum/StatefulPartitionedCall_2:output:0s_function_42044s_function_42046*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413372&
$s_function/StatefulPartitionedCall_2Ł
IdentityIdentity-s_function/StatefulPartitionedCall_2:output:0#^s_function/StatefulPartitionedCall%^s_function/StatefulPartitionedCall_1%^s_function/StatefulPartitionedCall_2"^shift_sns/StatefulPartitionedCall$^shift_sns/StatefulPartitionedCall_1$^shift_sns/StatefulPartitionedCall_2%^weighted_sum/StatefulPartitionedCall'^weighted_sum/StatefulPartitionedCall_1'^weighted_sum/StatefulPartitionedCall_2*
T0*#
_output_shapes
:’’’’’’’’’2

IdentityŻ

Identity_1Identity-s_function/StatefulPartitionedCall_1:output:0#^s_function/StatefulPartitionedCall%^s_function/StatefulPartitionedCall_1%^s_function/StatefulPartitionedCall_2"^shift_sns/StatefulPartitionedCall$^shift_sns/StatefulPartitionedCall_1$^shift_sns/StatefulPartitionedCall_2%^weighted_sum/StatefulPartitionedCall'^weighted_sum/StatefulPartitionedCall_1'^weighted_sum/StatefulPartitionedCall_2*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_1Ū

Identity_2Identity+s_function/StatefulPartitionedCall:output:0#^s_function/StatefulPartitionedCall%^s_function/StatefulPartitionedCall_1%^s_function/StatefulPartitionedCall_2"^shift_sns/StatefulPartitionedCall$^shift_sns/StatefulPartitionedCall_1$^shift_sns/StatefulPartitionedCall_2%^weighted_sum/StatefulPartitionedCall'^weighted_sum/StatefulPartitionedCall_1'^weighted_sum/StatefulPartitionedCall_2*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’: : : : 2H
"s_function/StatefulPartitionedCall"s_function/StatefulPartitionedCall2L
$s_function/StatefulPartitionedCall_1$s_function/StatefulPartitionedCall_12L
$s_function/StatefulPartitionedCall_2$s_function/StatefulPartitionedCall_22F
!shift_sns/StatefulPartitionedCall!shift_sns/StatefulPartitionedCall2J
#shift_sns/StatefulPartitionedCall_1#shift_sns/StatefulPartitionedCall_12J
#shift_sns/StatefulPartitionedCall_2#shift_sns/StatefulPartitionedCall_22L
$weighted_sum/StatefulPartitionedCall$weighted_sum/StatefulPartitionedCall2P
&weighted_sum/StatefulPartitionedCall_1&weighted_sum/StatefulPartitionedCall_12P
&weighted_sum/StatefulPartitionedCall_2&weighted_sum/StatefulPartitionedCall_2:Y U
/
_output_shapes
:’’’’’’’’’` 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:’’’’’’’’’` 
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:’’’’’’’’’` 
"
_user_specified_name
inputs/2:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/3
Ģ
x
(__inference_restored_function_body_41321

inputs
unknown:`
identity¢StatefulPartitionedCallŹ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_weighted_sum_layer_call_and_return_conditional_losses_387252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:’’’’’’’’’`: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’`
 
_user_specified_nameinputs
§
{
C__inference_any_of_3_layer_call_and_return_conditional_losses_38338

inputs
inputs_1
inputs_2
identityS
addAddV2inputsinputs_1*
T0*#
_output_shapes
:’’’’’’’’’2
addX
add_1AddV2add:z:0inputs_2*
T0*#
_output_shapes
:’’’’’’’’’2
add_1Q
mulMulinputsinputs_1*
T0*#
_output_shapes
:’’’’’’’’’2
mulW
mul_1Mulinputs_1inputs_2*
T0*#
_output_shapes
:’’’’’’’’’2
mul_1Y
add_2AddV2mul:z:0	mul_1:z:0*
T0*#
_output_shapes
:’’’’’’’’’2
add_2U
mul_2Mulinputs_2inputs*
T0*#
_output_shapes
:’’’’’’’’’2
mul_2[
add_3AddV2	add_2:z:0	mul_2:z:0*
T0*#
_output_shapes
:’’’’’’’’’2
add_3U
subSub	add_1:z:0	add_3:z:0*
T0*#
_output_shapes
:’’’’’’’’’2
subU
mul_3Mulinputsinputs_1*
T0*#
_output_shapes
:’’’’’’’’’2
mul_3X
mul_4Mul	mul_3:z:0inputs_2*
T0*#
_output_shapes
:’’’’’’’’’2
mul_4Y
add_4AddV2sub:z:0	mul_4:z:0*
T0*#
_output_shapes
:’’’’’’’’’2
add_4w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value/Minimum/y
clip_by_value/MinimumMinimum	add_4:z:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:’’’’’’’’’2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:’’’’’’’’’2
clip_by_valuea
IdentityIdentityclip_by_value:z:0*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:K G
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:KG
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:KG
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
É
x
(__inference_restored_function_body_41294

inputs
unknown:`
identity¢StatefulPartitionedCallĖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_shift_sns_layer_call_and_return_conditional_losses_390562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:’’’’’’’’’: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ō
„
M__inference_classifier3b_dummy_layer_call_and_return_conditional_losses_39541

inputs
inputs_1
inputs_2
identity

identity_1

identity_2V
IdentityIdentityinputs*
T0*#
_output_shapes
:’’’’’’’’’2

Identity\

Identity_1Identityinputs_1*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_1\

Identity_2Identityinputs_2*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:K G
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:KG
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:KG
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

Ŗ
,__inference_classifier3b_layer_call_fn_41554
classifier3b_bm1
classifier3b_bm2
classifier3b_bm3
classifier3b_rs
unknown:`
	unknown_0:`
	unknown_1: 
	unknown_2: 
identity

identity_1

identity_2¢StatefulPartitionedCallņ
StatefulPartitionedCallStatefulPartitionedCallclassifier3b_bm1classifier3b_bm2classifier3b_bm3classifier3b_rsunknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_classifier3b_layer_call_and_return_conditional_losses_415192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:’’’’’’’’’2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:’’’’’’’’’` 
*
_user_specified_nameclassifier3b/bm1:a]
/
_output_shapes
:’’’’’’’’’` 
*
_user_specified_nameclassifier3b/bm2:a]
/
_output_shapes
:’’’’’’’’’` 
*
_user_specified_nameclassifier3b/bm3:TP
#
_output_shapes
:’’’’’’’’’
)
_user_specified_nameclassifier3b/rs
į
Ķ
*__inference_physicalnn_layer_call_fn_41838
inputs_0
inputs_1
unknown:`
	unknown_0:`
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_physicalnn_layer_call_and_return_conditional_losses_417212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:’’’’’’’’’``:’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:’’’’’’’’’``
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/1


G__inference_weighted_sum_layer_call_and_return_conditional_losses_38725

inputs%
readvariableop_resource:`
identity¢ReadVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype02
ReadVariableOpZ
SoftmaxSoftmaxReadVariableOp:value:0*
T0*
_output_shapes
:`2	
Softmax{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2’
strided_sliceStridedSliceSoftmax:softmax:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:`*
ellipsis_mask*
new_axis_mask2
strided_slicec
mulMulstrided_slice:output:0inputs*
T0*'
_output_shapes
:’’’’’’’’’`2
mulx
Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2
Sum/reduction_indicesh
SumSummul:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:’’’’’’’’’2
Summ
IdentityIdentitySum:output:0^ReadVariableOp*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’`:2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:’’’’’’’’’`
 
_user_specified_nameinputs
»X
ņ
E__inference_physicalnn_layer_call_and_return_conditional_losses_41932
inputs_0
inputs_1*
classifier3b_shift_sns_41894:`-
classifier3b_weighted_sum_41909:`'
classifier3b_s_function_41916: '
classifier3b_s_function_41918: 
identity¢/classifier3b/s_function/StatefulPartitionedCall¢1classifier3b/s_function/StatefulPartitionedCall_1¢1classifier3b/s_function/StatefulPartitionedCall_2¢.classifier3b/shift_sns/StatefulPartitionedCall¢0classifier3b/shift_sns/StatefulPartitionedCall_1¢0classifier3b/shift_sns/StatefulPartitionedCall_2¢1classifier3b/weighted_sum/StatefulPartitionedCall¢3classifier3b/weighted_sum/StatefulPartitionedCall_1¢3classifier3b/weighted_sum/StatefulPartitionedCall_2Ū
#insert_channel_axis/PartitionedCallPartitionedCallinputs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’``* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412682%
#insert_channel_axis/PartitionedCall
split/PartitionedCallPartitionedCall,insert_channel_axis/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’` * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412782
split/PartitionedCallķ
%classifier3b/spectrum/PartitionedCallPartitionedCallsplit/PartitionedCall:output:2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412862'
%classifier3b/spectrum/PartitionedCall
.classifier3b/shift_sns/StatefulPartitionedCallStatefulPartitionedCallinputs_1classifier3b_shift_sns_41894*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_4129420
.classifier3b/shift_sns/StatefulPartitionedCallń
'classifier3b/spectrum/PartitionedCall_1PartitionedCallsplit/PartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412862)
'classifier3b/spectrum/PartitionedCall_1
0classifier3b/shift_sns/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1classifier3b_shift_sns_41894*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_4129422
0classifier3b/shift_sns/StatefulPartitionedCall_1ń
'classifier3b/spectrum/PartitionedCall_2PartitionedCallsplit/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412862)
'classifier3b/spectrum/PartitionedCall_2
0classifier3b/shift_sns/StatefulPartitionedCall_2StatefulPartitionedCallinputs_1classifier3b_shift_sns_41894*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_4129422
0classifier3b/shift_sns/StatefulPartitionedCall_2Ų
classifier3b/subtract/subSub.classifier3b/spectrum/PartitionedCall:output:07classifier3b/shift_sns/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2
classifier3b/subtract/subą
classifier3b/subtract/sub_1Sub0classifier3b/spectrum/PartitionedCall_1:output:09classifier3b/shift_sns/StatefulPartitionedCall_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2
classifier3b/subtract/sub_1ą
classifier3b/subtract/sub_2Sub0classifier3b/spectrum/PartitionedCall_2:output:09classifier3b/shift_sns/StatefulPartitionedCall_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2
classifier3b/subtract/sub_2ö
*classifier3b/relu_residual/PartitionedCallPartitionedCallclassifier3b/subtract/sub:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413112,
*classifier3b/relu_residual/PartitionedCallü
,classifier3b/relu_residual/PartitionedCall_1PartitionedCallclassifier3b/subtract/sub_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413112.
,classifier3b/relu_residual/PartitionedCall_1ü
,classifier3b/relu_residual/PartitionedCall_2PartitionedCallclassifier3b/subtract/sub_2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413112.
,classifier3b/relu_residual/PartitionedCall_2Ć
1classifier3b/weighted_sum/StatefulPartitionedCallStatefulPartitionedCall3classifier3b/relu_residual/PartitionedCall:output:0classifier3b_weighted_sum_41909*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_4132123
1classifier3b/weighted_sum/StatefulPartitionedCallÉ
3classifier3b/weighted_sum/StatefulPartitionedCall_1StatefulPartitionedCall5classifier3b/relu_residual/PartitionedCall_1:output:0classifier3b_weighted_sum_41909*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_4132125
3classifier3b/weighted_sum/StatefulPartitionedCall_1É
3classifier3b/weighted_sum/StatefulPartitionedCall_2StatefulPartitionedCall5classifier3b/relu_residual/PartitionedCall_2:output:0classifier3b_weighted_sum_41909*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_4132125
3classifier3b/weighted_sum/StatefulPartitionedCall_2å
/classifier3b/s_function/StatefulPartitionedCallStatefulPartitionedCall:classifier3b/weighted_sum/StatefulPartitionedCall:output:0classifier3b_s_function_41916classifier3b_s_function_41918*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_4133721
/classifier3b/s_function/StatefulPartitionedCallė
1classifier3b/s_function/StatefulPartitionedCall_1StatefulPartitionedCall<classifier3b/weighted_sum/StatefulPartitionedCall_1:output:0classifier3b_s_function_41916classifier3b_s_function_41918*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_4133723
1classifier3b/s_function/StatefulPartitionedCall_1ė
1classifier3b/s_function/StatefulPartitionedCall_2StatefulPartitionedCall<classifier3b/weighted_sum/StatefulPartitionedCall_2:output:0classifier3b_s_function_41916classifier3b_s_function_41918*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_4133723
1classifier3b/s_function/StatefulPartitionedCall_2
"classifier3b_dummy/PartitionedCallPartitionedCall:classifier3b/s_function/StatefulPartitionedCall_2:output:0:classifier3b/s_function/StatefulPartitionedCall_1:output:08classifier3b/s_function/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413592$
"classifier3b_dummy/PartitionedCallø
any_of_3/PartitionedCallPartitionedCall+classifier3b_dummy/PartitionedCall:output:0+classifier3b_dummy/PartitionedCall:output:1+classifier3b_dummy/PartitionedCall:output:2*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413692
any_of_3/PartitionedCallĀ
IdentityIdentity!any_of_3/PartitionedCall:output:00^classifier3b/s_function/StatefulPartitionedCall2^classifier3b/s_function/StatefulPartitionedCall_12^classifier3b/s_function/StatefulPartitionedCall_2/^classifier3b/shift_sns/StatefulPartitionedCall1^classifier3b/shift_sns/StatefulPartitionedCall_11^classifier3b/shift_sns/StatefulPartitionedCall_22^classifier3b/weighted_sum/StatefulPartitionedCall4^classifier3b/weighted_sum/StatefulPartitionedCall_14^classifier3b/weighted_sum/StatefulPartitionedCall_2*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:’’’’’’’’’``:’’’’’’’’’: : : : 2b
/classifier3b/s_function/StatefulPartitionedCall/classifier3b/s_function/StatefulPartitionedCall2f
1classifier3b/s_function/StatefulPartitionedCall_11classifier3b/s_function/StatefulPartitionedCall_12f
1classifier3b/s_function/StatefulPartitionedCall_21classifier3b/s_function/StatefulPartitionedCall_22`
.classifier3b/shift_sns/StatefulPartitionedCall.classifier3b/shift_sns/StatefulPartitionedCall2d
0classifier3b/shift_sns/StatefulPartitionedCall_10classifier3b/shift_sns/StatefulPartitionedCall_12d
0classifier3b/shift_sns/StatefulPartitionedCall_20classifier3b/shift_sns/StatefulPartitionedCall_22f
1classifier3b/weighted_sum/StatefulPartitionedCall1classifier3b/weighted_sum/StatefulPartitionedCall2j
3classifier3b/weighted_sum/StatefulPartitionedCall_13classifier3b/weighted_sum/StatefulPartitionedCall_12j
3classifier3b/weighted_sum/StatefulPartitionedCall_23classifier3b/weighted_sum/StatefulPartitionedCall_2:U Q
+
_output_shapes
:’’’’’’’’’``
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/1



2__inference_classifier3b_dummy_layer_call_fn_39552
inputs_0
inputs_1
inputs_2
identity

identity_1

identity_2
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_classifier3b_dummy_layer_call_and_return_conditional_losses_395412
PartitionedCallh
IdentityIdentityPartitionedCall:output:0*
T0*#
_output_shapes
:’’’’’’’’’2

Identityl

Identity_1IdentityPartitionedCall:output:1*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_1l

Identity_2IdentityPartitionedCall:output:2*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:M I
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/1:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/2
ų
`
(__inference_restored_function_body_41369

inputs
inputs_1
inputs_2
identity·
PartitionedCallPartitionedCallinputsinputs_1inputs_2*
Tin
2*
Tout
2*#
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_any_of_3_layer_call_and_return_conditional_losses_407662
PartitionedCallh
IdentityIdentityPartitionedCall:output:0*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:K G
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:KG
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:KG
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ķ

D__inference_shift_sns_layer_call_and_return_conditional_losses_40054

inputs
readvariableop_resource
identity¢ReadVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:`*
ellipsis_mask*
new_axis_mask2
strided_sliceG
LogLoginputs*
T0*#
_output_shapes
:’’’’’’’’’2
LogS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
ConstF
Log_1LogConst:output:0*
T0*
_output_shapes
: 2
Log_1_
truedivRealDivLog:y:0	Log_1:y:0*
T0*#
_output_shapes
:’’’’’’’’’2	
truediv
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSlicetruediv:z:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
ellipsis_mask*
new_axis_mask2
strided_slice_1S
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  \B2
mul/xm
mulMulmul/x:output:0strided_slice_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
mulf
addAddV2strided_slice:output:0mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’`2
addl
IdentityIdentityadd:z:0^ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’:2 
ReadVariableOpReadVariableOp:K G
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs


*__inference_s_function_layer_call_fn_38711

inputs
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallō
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_s_function_layer_call_and_return_conditional_losses_387042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

j
N__inference_insert_channel_axis_layer_call_and_return_conditional_losses_40532

inputs
identityk
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*/
_output_shapes
:’’’’’’’’’``2

ExpandDimso
IdentityIdentityExpandDims:output:0*
T0*/
_output_shapes
:’’’’’’’’’``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’``:S O
+
_output_shapes
:’’’’’’’’’``
 
_user_specified_nameinputs
µ

E__inference_physicalnn_layer_call_and_return_conditional_losses_41721

inputs
inputs_1 
classifier3b_41705:` 
classifier3b_41707:`
classifier3b_41709: 
classifier3b_41711: 
identity¢$classifier3b/StatefulPartitionedCallŁ
#insert_channel_axis/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’``* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412682%
#insert_channel_axis/PartitionedCall
split/PartitionedCallPartitionedCall,insert_channel_axis/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’` * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412782
split/PartitionedCallÕ
$classifier3b/StatefulPartitionedCallStatefulPartitionedCallsplit/PartitionedCall:output:0split/PartitionedCall:output:1split/PartitionedCall:output:2inputs_1classifier3b_41705classifier3b_41707classifier3b_41709classifier3b_41711*
Tin

2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_classifier3b_layer_call_and_return_conditional_losses_415192&
$classifier3b/StatefulPartitionedCallņ
"classifier3b_dummy/PartitionedCallPartitionedCall-classifier3b/StatefulPartitionedCall:output:0-classifier3b/StatefulPartitionedCall:output:1-classifier3b/StatefulPartitionedCall:output:2*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413592$
"classifier3b_dummy/PartitionedCallø
any_of_3/PartitionedCallPartitionedCall+classifier3b_dummy/PartitionedCall:output:0+classifier3b_dummy/PartitionedCall:output:1+classifier3b_dummy/PartitionedCall:output:2*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413692
any_of_3/PartitionedCall
IdentityIdentity!any_of_3/PartitionedCall:output:0%^classifier3b/StatefulPartitionedCall*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:’’’’’’’’’``:’’’’’’’’’: : : : 2L
$classifier3b/StatefulPartitionedCall$classifier3b/StatefulPartitionedCall:S O
+
_output_shapes
:’’’’’’’’’``
 
_user_specified_nameinputs:KG
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
»X
ņ
E__inference_physicalnn_layer_call_and_return_conditional_losses_41885
inputs_0
inputs_1*
classifier3b_shift_sns_41847:`-
classifier3b_weighted_sum_41862:`'
classifier3b_s_function_41869: '
classifier3b_s_function_41871: 
identity¢/classifier3b/s_function/StatefulPartitionedCall¢1classifier3b/s_function/StatefulPartitionedCall_1¢1classifier3b/s_function/StatefulPartitionedCall_2¢.classifier3b/shift_sns/StatefulPartitionedCall¢0classifier3b/shift_sns/StatefulPartitionedCall_1¢0classifier3b/shift_sns/StatefulPartitionedCall_2¢1classifier3b/weighted_sum/StatefulPartitionedCall¢3classifier3b/weighted_sum/StatefulPartitionedCall_1¢3classifier3b/weighted_sum/StatefulPartitionedCall_2Ū
#insert_channel_axis/PartitionedCallPartitionedCallinputs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’``* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412682%
#insert_channel_axis/PartitionedCall
split/PartitionedCallPartitionedCall,insert_channel_axis/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’` * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412782
split/PartitionedCallķ
%classifier3b/spectrum/PartitionedCallPartitionedCallsplit/PartitionedCall:output:2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412862'
%classifier3b/spectrum/PartitionedCall
.classifier3b/shift_sns/StatefulPartitionedCallStatefulPartitionedCallinputs_1classifier3b_shift_sns_41847*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_4129420
.classifier3b/shift_sns/StatefulPartitionedCallń
'classifier3b/spectrum/PartitionedCall_1PartitionedCallsplit/PartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412862)
'classifier3b/spectrum/PartitionedCall_1
0classifier3b/shift_sns/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1classifier3b_shift_sns_41847*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_4129422
0classifier3b/shift_sns/StatefulPartitionedCall_1ń
'classifier3b/spectrum/PartitionedCall_2PartitionedCallsplit/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412862)
'classifier3b/spectrum/PartitionedCall_2
0classifier3b/shift_sns/StatefulPartitionedCall_2StatefulPartitionedCallinputs_1classifier3b_shift_sns_41847*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_4129422
0classifier3b/shift_sns/StatefulPartitionedCall_2Ų
classifier3b/subtract/subSub.classifier3b/spectrum/PartitionedCall:output:07classifier3b/shift_sns/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2
classifier3b/subtract/subą
classifier3b/subtract/sub_1Sub0classifier3b/spectrum/PartitionedCall_1:output:09classifier3b/shift_sns/StatefulPartitionedCall_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2
classifier3b/subtract/sub_1ą
classifier3b/subtract/sub_2Sub0classifier3b/spectrum/PartitionedCall_2:output:09classifier3b/shift_sns/StatefulPartitionedCall_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2
classifier3b/subtract/sub_2ö
*classifier3b/relu_residual/PartitionedCallPartitionedCallclassifier3b/subtract/sub:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413112,
*classifier3b/relu_residual/PartitionedCallü
,classifier3b/relu_residual/PartitionedCall_1PartitionedCallclassifier3b/subtract/sub_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413112.
,classifier3b/relu_residual/PartitionedCall_1ü
,classifier3b/relu_residual/PartitionedCall_2PartitionedCallclassifier3b/subtract/sub_2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413112.
,classifier3b/relu_residual/PartitionedCall_2Ć
1classifier3b/weighted_sum/StatefulPartitionedCallStatefulPartitionedCall3classifier3b/relu_residual/PartitionedCall:output:0classifier3b_weighted_sum_41862*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_4132123
1classifier3b/weighted_sum/StatefulPartitionedCallÉ
3classifier3b/weighted_sum/StatefulPartitionedCall_1StatefulPartitionedCall5classifier3b/relu_residual/PartitionedCall_1:output:0classifier3b_weighted_sum_41862*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_4132125
3classifier3b/weighted_sum/StatefulPartitionedCall_1É
3classifier3b/weighted_sum/StatefulPartitionedCall_2StatefulPartitionedCall5classifier3b/relu_residual/PartitionedCall_2:output:0classifier3b_weighted_sum_41862*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_4132125
3classifier3b/weighted_sum/StatefulPartitionedCall_2å
/classifier3b/s_function/StatefulPartitionedCallStatefulPartitionedCall:classifier3b/weighted_sum/StatefulPartitionedCall:output:0classifier3b_s_function_41869classifier3b_s_function_41871*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_4133721
/classifier3b/s_function/StatefulPartitionedCallė
1classifier3b/s_function/StatefulPartitionedCall_1StatefulPartitionedCall<classifier3b/weighted_sum/StatefulPartitionedCall_1:output:0classifier3b_s_function_41869classifier3b_s_function_41871*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_4133723
1classifier3b/s_function/StatefulPartitionedCall_1ė
1classifier3b/s_function/StatefulPartitionedCall_2StatefulPartitionedCall<classifier3b/weighted_sum/StatefulPartitionedCall_2:output:0classifier3b_s_function_41869classifier3b_s_function_41871*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_4133723
1classifier3b/s_function/StatefulPartitionedCall_2
"classifier3b_dummy/PartitionedCallPartitionedCall:classifier3b/s_function/StatefulPartitionedCall_2:output:0:classifier3b/s_function/StatefulPartitionedCall_1:output:08classifier3b/s_function/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413592$
"classifier3b_dummy/PartitionedCallø
any_of_3/PartitionedCallPartitionedCall+classifier3b_dummy/PartitionedCall:output:0+classifier3b_dummy/PartitionedCall:output:1+classifier3b_dummy/PartitionedCall:output:2*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413692
any_of_3/PartitionedCallĀ
IdentityIdentity!any_of_3/PartitionedCall:output:00^classifier3b/s_function/StatefulPartitionedCall2^classifier3b/s_function/StatefulPartitionedCall_12^classifier3b/s_function/StatefulPartitionedCall_2/^classifier3b/shift_sns/StatefulPartitionedCall1^classifier3b/shift_sns/StatefulPartitionedCall_11^classifier3b/shift_sns/StatefulPartitionedCall_22^classifier3b/weighted_sum/StatefulPartitionedCall4^classifier3b/weighted_sum/StatefulPartitionedCall_14^classifier3b/weighted_sum/StatefulPartitionedCall_2*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:’’’’’’’’’``:’’’’’’’’’: : : : 2b
/classifier3b/s_function/StatefulPartitionedCall/classifier3b/s_function/StatefulPartitionedCall2f
1classifier3b/s_function/StatefulPartitionedCall_11classifier3b/s_function/StatefulPartitionedCall_12f
1classifier3b/s_function/StatefulPartitionedCall_21classifier3b/s_function/StatefulPartitionedCall_22`
.classifier3b/shift_sns/StatefulPartitionedCall.classifier3b/shift_sns/StatefulPartitionedCall2d
0classifier3b/shift_sns/StatefulPartitionedCall_10classifier3b/shift_sns/StatefulPartitionedCall_12d
0classifier3b/shift_sns/StatefulPartitionedCall_20classifier3b/shift_sns/StatefulPartitionedCall_22f
1classifier3b/weighted_sum/StatefulPartitionedCall1classifier3b/weighted_sum/StatefulPartitionedCall2j
3classifier3b/weighted_sum/StatefulPartitionedCall_13classifier3b/weighted_sum/StatefulPartitionedCall_12j
3classifier3b/weighted_sum/StatefulPartitionedCall_23classifier3b/weighted_sum/StatefulPartitionedCall_2:U Q
+
_output_shapes
:’’’’’’’’’``
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/1
÷

D__inference_shift_sns_layer_call_and_return_conditional_losses_39056

inputs%
readvariableop_resource:`
identity¢ReadVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:`*
ellipsis_mask*
new_axis_mask2
strided_sliceG
LogLoginputs*
T0*#
_output_shapes
:’’’’’’’’’2
LogS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
ConstF
Log_1LogConst:output:0*
T0*
_output_shapes
: 2
Log_1_
truedivRealDivLog:y:0	Log_1:y:0*
T0*#
_output_shapes
:’’’’’’’’’2	
truediv
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSlicetruediv:z:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
ellipsis_mask*
new_axis_mask2
strided_slice_1S
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  \B2
mul/xm
mulMulmul/x:output:0strided_slice_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
mulf
addAddV2strided_slice:output:0mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’`2
addl
IdentityIdentityadd:z:0^ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’:2 
ReadVariableOpReadVariableOp:K G
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ķ
d
(__inference_restored_function_body_41278

inputs
identity

identity_1

identity_2ā
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*e
_output_shapesS
Q:’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’` * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_split_layer_call_and_return_conditional_losses_395972
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:’’’’’’’’’` 2

Identityx

Identity_1IdentityPartitionedCall:output:1*
T0*/
_output_shapes
:’’’’’’’’’` 2

Identity_1x

Identity_2IdentityPartitionedCall:output:2*
T0*/
_output_shapes
:’’’’’’’’’` 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’``:W S
/
_output_shapes
:’’’’’’’’’``
 
_user_specified_nameinputs
å	

(__inference_restored_function_body_41359

inputs
inputs_1
inputs_2
identity

identity_1

identity_2į
PartitionedCallPartitionedCallinputsinputs_1inputs_2*
Tin
2*
Tout
2*A
_output_shapes/
-:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_classifier3b_dummy_layer_call_and_return_conditional_losses_395862
PartitionedCallh
IdentityIdentityPartitionedCall:output:0*
T0*#
_output_shapes
:’’’’’’’’’2

Identityl

Identity_1IdentityPartitionedCall:output:1*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_1l

Identity_2IdentityPartitionedCall:output:2*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:K G
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:KG
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:KG
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
»Q
¦
G__inference_classifier3b_layer_call_and_return_conditional_losses_41640
classifier3b_bm1
classifier3b_bm2
classifier3b_bm3
classifier3b_rs
shift_sns_41604:` 
weighted_sum_41619:`
s_function_41626: 
s_function_41628: 
identity

identity_1

identity_2¢"s_function/StatefulPartitionedCall¢$s_function/StatefulPartitionedCall_1¢$s_function/StatefulPartitionedCall_2¢!shift_sns/StatefulPartitionedCall¢#shift_sns/StatefulPartitionedCall_1¢#shift_sns/StatefulPartitionedCall_2¢$weighted_sum/StatefulPartitionedCall¢&weighted_sum/StatefulPartitionedCall_1¢&weighted_sum/StatefulPartitionedCall_2Å
spectrum/PartitionedCallPartitionedCallclassifier3b_bm3*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412862
spectrum/PartitionedCalló
!shift_sns/StatefulPartitionedCallStatefulPartitionedCallclassifier3b_rsshift_sns_41604*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412942#
!shift_sns/StatefulPartitionedCallÉ
spectrum/PartitionedCall_1PartitionedCallclassifier3b_bm2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412862
spectrum/PartitionedCall_1÷
#shift_sns/StatefulPartitionedCall_1StatefulPartitionedCallclassifier3b_rsshift_sns_41604*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412942%
#shift_sns/StatefulPartitionedCall_1É
spectrum/PartitionedCall_2PartitionedCallclassifier3b_bm1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412862
spectrum/PartitionedCall_2÷
#shift_sns/StatefulPartitionedCall_2StatefulPartitionedCallclassifier3b_rsshift_sns_41604*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_412942%
#shift_sns/StatefulPartitionedCall_2
subtract/PartitionedCallPartitionedCall!spectrum/PartitionedCall:output:0*shift_sns/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_subtract_layer_call_and_return_conditional_losses_414012
subtract/PartitionedCall¦
subtract/PartitionedCall_1PartitionedCall#spectrum/PartitionedCall_1:output:0,shift_sns/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_subtract_layer_call_and_return_conditional_losses_414012
subtract/PartitionedCall_1¦
subtract/PartitionedCall_2PartitionedCall#spectrum/PartitionedCall_2:output:0,shift_sns/StatefulPartitionedCall_2:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_subtract_layer_call_and_return_conditional_losses_414012
subtract/PartitionedCall_2ą
relu_residual/PartitionedCallPartitionedCall!subtract/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413112
relu_residual/PartitionedCallę
relu_residual/PartitionedCall_1PartitionedCall#subtract/PartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413112!
relu_residual/PartitionedCall_1ę
relu_residual/PartitionedCall_2PartitionedCall#subtract/PartitionedCall_2:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413112!
relu_residual/PartitionedCall_2
$weighted_sum/StatefulPartitionedCallStatefulPartitionedCall&relu_residual/PartitionedCall:output:0weighted_sum_41619*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413212&
$weighted_sum/StatefulPartitionedCall
&weighted_sum/StatefulPartitionedCall_1StatefulPartitionedCall(relu_residual/PartitionedCall_1:output:0weighted_sum_41619*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413212(
&weighted_sum/StatefulPartitionedCall_1
&weighted_sum/StatefulPartitionedCall_2StatefulPartitionedCall(relu_residual/PartitionedCall_2:output:0weighted_sum_41619*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413212(
&weighted_sum/StatefulPartitionedCall_2¤
"s_function/StatefulPartitionedCallStatefulPartitionedCall-weighted_sum/StatefulPartitionedCall:output:0s_function_41626s_function_41628*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413372$
"s_function/StatefulPartitionedCallŖ
$s_function/StatefulPartitionedCall_1StatefulPartitionedCall/weighted_sum/StatefulPartitionedCall_1:output:0s_function_41626s_function_41628*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413372&
$s_function/StatefulPartitionedCall_1Ŗ
$s_function/StatefulPartitionedCall_2StatefulPartitionedCall/weighted_sum/StatefulPartitionedCall_2:output:0s_function_41626s_function_41628*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_413372&
$s_function/StatefulPartitionedCall_2Ł
IdentityIdentity-s_function/StatefulPartitionedCall_2:output:0#^s_function/StatefulPartitionedCall%^s_function/StatefulPartitionedCall_1%^s_function/StatefulPartitionedCall_2"^shift_sns/StatefulPartitionedCall$^shift_sns/StatefulPartitionedCall_1$^shift_sns/StatefulPartitionedCall_2%^weighted_sum/StatefulPartitionedCall'^weighted_sum/StatefulPartitionedCall_1'^weighted_sum/StatefulPartitionedCall_2*
T0*#
_output_shapes
:’’’’’’’’’2

IdentityŻ

Identity_1Identity-s_function/StatefulPartitionedCall_1:output:0#^s_function/StatefulPartitionedCall%^s_function/StatefulPartitionedCall_1%^s_function/StatefulPartitionedCall_2"^shift_sns/StatefulPartitionedCall$^shift_sns/StatefulPartitionedCall_1$^shift_sns/StatefulPartitionedCall_2%^weighted_sum/StatefulPartitionedCall'^weighted_sum/StatefulPartitionedCall_1'^weighted_sum/StatefulPartitionedCall_2*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_1Ū

Identity_2Identity+s_function/StatefulPartitionedCall:output:0#^s_function/StatefulPartitionedCall%^s_function/StatefulPartitionedCall_1%^s_function/StatefulPartitionedCall_2"^shift_sns/StatefulPartitionedCall$^shift_sns/StatefulPartitionedCall_1$^shift_sns/StatefulPartitionedCall_2%^weighted_sum/StatefulPartitionedCall'^weighted_sum/StatefulPartitionedCall_1'^weighted_sum/StatefulPartitionedCall_2*
T0*#
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’: : : : 2H
"s_function/StatefulPartitionedCall"s_function/StatefulPartitionedCall2L
$s_function/StatefulPartitionedCall_1$s_function/StatefulPartitionedCall_12L
$s_function/StatefulPartitionedCall_2$s_function/StatefulPartitionedCall_22F
!shift_sns/StatefulPartitionedCall!shift_sns/StatefulPartitionedCall2J
#shift_sns/StatefulPartitionedCall_1#shift_sns/StatefulPartitionedCall_12J
#shift_sns/StatefulPartitionedCall_2#shift_sns/StatefulPartitionedCall_22L
$weighted_sum/StatefulPartitionedCall$weighted_sum/StatefulPartitionedCall2P
&weighted_sum/StatefulPartitionedCall_1&weighted_sum/StatefulPartitionedCall_12P
&weighted_sum/StatefulPartitionedCall_2&weighted_sum/StatefulPartitionedCall_2:a ]
/
_output_shapes
:’’’’’’’’’` 
*
_user_specified_nameclassifier3b/bm1:a]
/
_output_shapes
:’’’’’’’’’` 
*
_user_specified_nameclassifier3b/bm2:a]
/
_output_shapes
:’’’’’’’’’` 
*
_user_specified_nameclassifier3b/bm3:TP
#
_output_shapes
:’’’’’’’’’
)
_user_specified_nameclassifier3b/rs
ą
|
@__inference_split_layer_call_and_return_conditional_losses_39597

inputs
identity

identity_1

identity_2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ž’’’’’’’’2
split/split_dimŗ
splitSplitsplit/split_dim:output:0inputs*
T0*e
_output_shapesS
Q:’’’’’’’’’` :’’’’’’’’’` :’’’’’’’’’` *
	num_split2
splitj
IdentityIdentitysplit:output:0*
T0*/
_output_shapes
:’’’’’’’’’` 2

Identityn

Identity_1Identitysplit:output:1*
T0*/
_output_shapes
:’’’’’’’’’` 2

Identity_1n

Identity_2Identitysplit:output:2*
T0*/
_output_shapes
:’’’’’’’’’` 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’``:W S
/
_output_shapes
:’’’’’’’’’``
 
_user_specified_nameinputs
Ø5
ä
!__inference__traced_restore_42176
file_prefix,
assignvariableop_shift_sns_sns:`5
'assignvariableop_1_weighted_sum_weights:`-
#assignvariableop_2_s_function_alpha: ,
"assignvariableop_3_s_function_beta: "
assignvariableop_4_total: "
assignvariableop_5_count: $
assignvariableop_6_total_1: $
assignvariableop_7_count_1: /
!assignvariableop_8_true_positives:0
"assignvariableop_9_false_negatives:2
$assignvariableop_10_true_positives_1:1
#assignvariableop_11_false_positives:
identity_13¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*©
valueBB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesØ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesģ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_shift_sns_snsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¬
AssignVariableOp_1AssignVariableOp'assignvariableop_1_weighted_sum_weightsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ø
AssignVariableOp_2AssignVariableOp#assignvariableop_2_s_function_alphaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3§
AssignVariableOp_3AssignVariableOp"assignvariableop_3_s_function_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4
AssignVariableOp_4AssignVariableOpassignvariableop_4_totalIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5
AssignVariableOp_5AssignVariableOpassignvariableop_5_countIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOpassignvariableop_6_total_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOpassignvariableop_7_count_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¦
AssignVariableOp_8AssignVariableOp!assignvariableop_8_true_positivesIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9§
AssignVariableOp_9AssignVariableOp"assignvariableop_9_false_negativesIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¬
AssignVariableOp_10AssignVariableOp$assignvariableop_10_true_positives_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11«
AssignVariableOp_11AssignVariableOp#assignvariableop_11_false_positivesIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpę
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12Ł
Identity_13IdentityIdentity_12:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_13"#
identity_13Identity_13:output:0*-
_input_shapes
: : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ć

(__inference_restored_function_body_41337

inputs
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*#
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_s_function_layer_call_and_return_conditional_losses_387362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
²#

__inference__traced_save_42130
file_prefix,
(savev2_shift_sns_sns_read_readvariableop3
/savev2_weighted_sum_weights_read_readvariableop/
+savev2_s_function_alpha_read_readvariableop.
*savev2_s_function_beta_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop-
)savev2_true_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop/
+savev2_true_positives_1_read_readvariableop.
*savev2_false_positives_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*©
valueBB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¢
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices¶
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_shift_sns_sns_read_readvariableop/savev2_weighted_sum_weights_read_readvariableop+savev2_s_function_alpha_read_readvariableop*savev2_s_function_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_negatives_read_readvariableop+savev2_true_positives_1_read_readvariableop*savev2_false_positives_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2ŗ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes”
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*G
_input_shapes6
4: :`:`: : : : : : ::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:`: 

_output_shapes
:`:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
”
b
(__inference_any_of_3_layer_call_fn_38345
inputs_0
inputs_1
inputs_2
identityŲ
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_any_of_3_layer_call_and_return_conditional_losses_383382
PartitionedCallh
IdentityIdentityPartitionedCall:output:0*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:M I
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/1:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/2
į
Ķ
*__inference_physicalnn_layer_call_fn_41680
input_sp
input_rs
unknown:`
	unknown_0:`
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_spinput_rsunknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_physicalnn_layer_call_and_return_conditional_losses_416692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:’’’’’’’’’``:’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:’’’’’’’’’``
"
_user_specified_name
input_sp:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
input_rs
Ż
Č
E__inference_s_function_layer_call_and_return_conditional_losses_38704

inputs
pow_readvariableop_resource
readvariableop_resource
identity¢ReadVariableOp¢pow/ReadVariableOp|
pow/ReadVariableOpReadVariableOppow_readvariableop_resource*
_output_shapes
: *
dtype02
pow/ReadVariableOpc
powPowinputspow/ReadVariableOp:value:0*
T0*#
_output_shapes
:’’’’’’’’’2
powp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp`
mulMulReadVariableOp:value:0pow:z:0*
T0*#
_output_shapes
:’’’’’’’’’2
mulK
TanhTanhmul:z:0*
T0*#
_output_shapes
:’’’’’’’’’2
Tanh~
IdentityIdentityTanh:y:0^ReadVariableOp^pow/ReadVariableOp*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’::2 
ReadVariableOpReadVariableOp2(
pow/ReadVariableOppow/ReadVariableOp:K G
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
į
Ķ
*__inference_physicalnn_layer_call_fn_41824
inputs_0
inputs_1
unknown:`
	unknown_0:`
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_physicalnn_layer_call_and_return_conditional_losses_416692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:’’’’’’’’’``:’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:’’’’’’’’’``
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/1"ĢL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*č
serving_defaultŌ
9
input_rs-
serving_default_input_rs:0’’’’’’’’’
A
input_sp5
serving_default_input_sp:0’’’’’’’’’``8
any_of_3,
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:¼±

layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer-5
layer-6
	optimizer
#	_self_saveable_object_factories

	variables
regularization_losses
trainable_variables
	keras_api

signatures
±_default_save_signature
²__call__
+³&call_and_return_all_conditional_losses"¢
_tf_keras_network{"name": "physicalnn", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "physicalnn", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 96]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_sp"}, "name": "input_sp", "inbound_nodes": []}, {"class_name": "InsertChannelAxis_", "config": {"name": "insert_channel_axis", "trainable": true, "dtype": "float32", "ch_axis": -1}, "name": "insert_channel_axis", "inbound_nodes": [[["input_sp", 0, 0, {}]]]}, {"class_name": "Split_", "config": {"name": "split", "trainable": true, "dtype": "float32", "axis": -3, "n_splits": 3}, "name": "split", "inbound_nodes": [[["insert_channel_axis", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_rs"}, "name": "input_rs", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "classifier3b", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "classifier3b/bm1"}, "name": "classifier3b/bm1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "classifier3b/rs"}, "name": "classifier3b/rs", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "classifier3b/bm2"}, "name": "classifier3b/bm2", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "classifier3b/bm3"}, "name": "classifier3b/bm3", "inbound_nodes": []}, {"class_name": "Spectrum_", "config": {"name": "spectrum", "trainable": true, "dtype": "float32", "t_axis": -2, "ch_axis": -1}, "name": "spectrum", "inbound_nodes": [[["classifier3b/bm1", 0, 0, {}]], [["classifier3b/bm2", 0, 0, {}]], [["classifier3b/bm3", 0, 0, {}]]]}, {"class_name": "ShiftSNS_", "config": {"name": "shift_sns", "trainable": true, "dtype": "float32", "sns_shape": [96], "cutin_freq": 4000, "isns": [-51.73952102661133, -51.91908264160156, -52.17738723754883, -52.491268157958984, -52.74644470214844, -53.01625061035156, -53.32057571411133, -53.5711784362793, -53.81317138671875, -54.05160903930664, -54.34244918823242, -54.645511627197266, -54.94461441040039, -55.16508483886719, -55.3690185546875, -55.55717468261719, -55.71791076660156, -55.86759948730469, -56.01735305786133, -56.17418670654297, -56.310523986816406, -56.43460464477539, -56.54563903808594, -56.64398956298828, -56.74330520629883, -56.84583282470703, -56.98292922973633, -57.1292724609375, -57.27627182006836, -57.38688659667969, -57.48052978515625, -57.5677375793457, -57.65999984741211, -57.707054138183594, -57.74354934692383, -57.785884857177734, -57.789222717285156, -57.77459716796875, -57.74294662475586, -57.74364471435547, -57.736480712890625, -57.722232818603516, -57.75682830810547, -57.78118133544922, -57.803714752197266, -57.85961151123047, -57.92207717895508, -57.99568176269531, -58.09674072265625, -58.18413543701172, -58.263145446777344, -58.332786560058594, -58.412906646728516, -58.507652282714844, -58.621131896972656, -58.73277282714844, -58.814598083496094, -58.86747360229492, -58.936519622802734, -58.98716735839844, -59.031463623046875, -59.116207122802734, -59.16674041748047, -59.193458557128906, -59.19350051879883, -59.217323303222656, -59.24707794189453, -59.274566650390625, -59.22399139404297, -59.18412780761719, -59.17753982543945, -59.254913330078125, -59.29663848876953, -59.29777145385742, -59.27923583984375, -59.28166580200195, -59.29579162597656, -59.292144775390625, -59.28715515136719, -59.2860107421875, -59.300048828125, -59.321205139160156, -59.35308837890625, -59.403472900390625, -59.459136962890625, -59.5334587097168, -59.63352966308594, -59.789154052734375, -60.00239181518555, -60.27350616455078, -60.605594635009766, -60.99026870727539, -61.40313720703125, -61.763999938964844, -62.23042297363281, -62.628910064697266], "F": [4042.10546875, 4134.2939453125, 4226.482421875, 4318.67041015625, 4410.85888671875, 4503.04736328125, 4595.23583984375, 4687.423828125, 4779.6123046875, 4871.80078125, 4963.9892578125, 5056.17724609375, 5148.36572265625, 5240.55419921875, 5332.74267578125, 5424.9306640625, 5517.119140625, 5609.3076171875, 5701.49609375, 5793.6845703125, 5885.87255859375, 5978.06103515625, 6070.24951171875, 6162.4375, 6254.6259765625, 6346.814453125, 6439.0029296875, 6531.19140625, 6623.3798828125, 6715.56787109375, 6807.75634765625, 6899.9443359375, 6992.1328125, 7084.3212890625, 7176.509765625, 7268.6982421875, 7360.88671875, 7453.0751953125, 7545.26318359375, 7637.45166015625, 7729.6396484375, 7821.828125, 7914.0166015625, 8006.205078125, 8098.3935546875, 8190.58154296875, 8282.76953125, 8374.958984375, 8467.146484375, 8559.3359375, 8651.5234375, 8743.7119140625, 8835.900390625, 8928.0888671875, 9020.27734375, 9112.46484375, 9204.654296875, 9296.841796875, 9389.0302734375, 9481.21875, 9573.4072265625, 9665.595703125, 9757.783203125, 9849.97265625, 9942.16015625, 10034.349609375, 10126.537109375, 10218.7255859375, 10310.9140625, 10403.1025390625, 10495.291015625, 10587.478515625, 10679.66796875, 10771.85546875, 10864.044921875, 10956.232421875, 11048.4208984375, 11140.609375, 11232.7978515625, 11324.986328125, 11417.173828125, 11509.36328125, 11601.55078125, 11693.740234375, 11785.927734375, 11878.1162109375, 11970.3046875, 12062.4931640625, 12154.681640625, 12246.8701171875, 12339.0576171875, 12431.24609375, 12523.4345703125, 12615.623046875, 12707.8115234375, 12800.0]}, "name": "shift_sns", "inbound_nodes": [[["classifier3b/rs", 0, 0, {}]], [["classifier3b/rs", 0, 0, {}]], [["classifier3b/rs", 0, 0, {}]]]}, {"class_name": "Subtract", "config": {"name": "subtract", "trainable": true, "dtype": "float32"}, "name": "subtract", "inbound_nodes": [[["spectrum", 0, 0, {}], ["shift_sns", 0, 0, {}]], [["spectrum", 1, 0, {}], ["shift_sns", 1, 0, {}]], [["spectrum", 2, 0, {}], ["shift_sns", 2, 0, {}]]]}, {"class_name": "ReLU_", "config": {"name": "relu_residual", "trainable": true, "dtype": "float32"}, "name": "relu_residual", "inbound_nodes": [[["subtract", 0, 0, {}]], [["subtract", 1, 0, {}]], [["subtract", 2, 0, {}]]]}, {"class_name": "WeightedSum_", "config": {"name": "weighted_sum", "trainable": true, "dtype": "float32"}, "name": "weighted_sum", "inbound_nodes": [[["relu_residual", 0, 0, {}]], [["relu_residual", 1, 0, {}]], [["relu_residual", 2, 0, {}]]]}, {"class_name": "SFunction_", "config": {"name": "s_function", "trainable": true, "dtype": "float32", "alpha": 0.02, "beta": 2}, "name": "s_function", "inbound_nodes": [[["weighted_sum", 0, 0, {}]], [["weighted_sum", 1, 0, {}]], [["weighted_sum", 2, 0, {}]]]}], "input_layers": [["classifier3b/bm1", 0, 0], ["classifier3b/bm2", 0, 0], ["classifier3b/bm3", 0, 0], ["classifier3b/rs", 0, 0]], "output_layers": [["s_function", 0, 0], ["s_function", 1, 0], ["s_function", 2, 0]]}, "name": "classifier3b", "inbound_nodes": [[["split", 0, 0, {}], ["split", 0, 1, {}], ["split", 0, 2, {}], ["input_rs", 0, 0, {}]]]}, {"class_name": "Dummy_", "config": {"name": "classifier3b_dummy", "trainable": true, "dtype": "float32"}, "name": "classifier3b_dummy", "inbound_nodes": [[["classifier3b", 1, 0, {}], ["classifier3b", 1, 1, {}], ["classifier3b", 1, 2, {}]]]}, {"class_name": "AnyOf3_", "config": {"name": "any_of_3", "trainable": true, "dtype": "float32"}, "name": "any_of_3", "inbound_nodes": [[["classifier3b_dummy", 0, 0, {}], ["classifier3b_dummy", 0, 1, {}], ["classifier3b_dummy", 0, 2, {}]]]}], "input_layers": [["input_sp", 0, 0], ["input_rs", 0, 0]], "output_layers": [["any_of_3", 0, 0]]}, "shared_object_id": 17, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 96, 96]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null]}, "ndim": 1, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 96, 96]}, {"class_name": "TensorShape", "items": [null]}], "is_graph_network": true, "save_spec": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 96, 96]}, "float32", "input_sp"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null]}, "float32", "input_rs"]}], "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "physicalnn", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 96]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_sp"}, "name": "input_sp", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InsertChannelAxis_", "config": {"name": "insert_channel_axis", "trainable": true, "dtype": "float32", "ch_axis": -1}, "name": "insert_channel_axis", "inbound_nodes": [[["input_sp", 0, 0, {}]]], "shared_object_id": 1}, {"class_name": "Split_", "config": {"name": "split", "trainable": true, "dtype": "float32", "axis": -3, "n_splits": 3}, "name": "split", "inbound_nodes": [[["insert_channel_axis", 0, 0, {}]]], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_rs"}, "name": "input_rs", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "Functional", "config": {"name": "classifier3b", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "classifier3b/bm1"}, "name": "classifier3b/bm1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "classifier3b/rs"}, "name": "classifier3b/rs", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "classifier3b/bm2"}, "name": "classifier3b/bm2", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "classifier3b/bm3"}, "name": "classifier3b/bm3", "inbound_nodes": []}, {"class_name": "Spectrum_", "config": {"name": "spectrum", "trainable": true, "dtype": "float32", "t_axis": -2, "ch_axis": -1}, "name": "spectrum", "inbound_nodes": [[["classifier3b/bm1", 0, 0, {}]], [["classifier3b/bm2", 0, 0, {}]], [["classifier3b/bm3", 0, 0, {}]]]}, {"class_name": "ShiftSNS_", "config": {"name": "shift_sns", "trainable": true, "dtype": "float32", "sns_shape": [96], "cutin_freq": 4000, "isns": [-51.73952102661133, -51.91908264160156, -52.17738723754883, -52.491268157958984, -52.74644470214844, -53.01625061035156, -53.32057571411133, -53.5711784362793, -53.81317138671875, -54.05160903930664, -54.34244918823242, -54.645511627197266, -54.94461441040039, -55.16508483886719, -55.3690185546875, -55.55717468261719, -55.71791076660156, -55.86759948730469, -56.01735305786133, -56.17418670654297, -56.310523986816406, -56.43460464477539, -56.54563903808594, -56.64398956298828, -56.74330520629883, -56.84583282470703, -56.98292922973633, -57.1292724609375, -57.27627182006836, -57.38688659667969, -57.48052978515625, -57.5677375793457, -57.65999984741211, -57.707054138183594, -57.74354934692383, -57.785884857177734, -57.789222717285156, -57.77459716796875, -57.74294662475586, -57.74364471435547, -57.736480712890625, -57.722232818603516, -57.75682830810547, -57.78118133544922, -57.803714752197266, -57.85961151123047, -57.92207717895508, -57.99568176269531, -58.09674072265625, -58.18413543701172, -58.263145446777344, -58.332786560058594, -58.412906646728516, -58.507652282714844, -58.621131896972656, -58.73277282714844, -58.814598083496094, -58.86747360229492, -58.936519622802734, -58.98716735839844, -59.031463623046875, -59.116207122802734, -59.16674041748047, -59.193458557128906, -59.19350051879883, -59.217323303222656, -59.24707794189453, -59.274566650390625, -59.22399139404297, -59.18412780761719, -59.17753982543945, -59.254913330078125, -59.29663848876953, -59.29777145385742, -59.27923583984375, -59.28166580200195, -59.29579162597656, -59.292144775390625, -59.28715515136719, -59.2860107421875, -59.300048828125, -59.321205139160156, -59.35308837890625, -59.403472900390625, -59.459136962890625, -59.5334587097168, -59.63352966308594, -59.789154052734375, -60.00239181518555, -60.27350616455078, -60.605594635009766, -60.99026870727539, -61.40313720703125, -61.763999938964844, -62.23042297363281, -62.628910064697266], "F": [4042.10546875, 4134.2939453125, 4226.482421875, 4318.67041015625, 4410.85888671875, 4503.04736328125, 4595.23583984375, 4687.423828125, 4779.6123046875, 4871.80078125, 4963.9892578125, 5056.17724609375, 5148.36572265625, 5240.55419921875, 5332.74267578125, 5424.9306640625, 5517.119140625, 5609.3076171875, 5701.49609375, 5793.6845703125, 5885.87255859375, 5978.06103515625, 6070.24951171875, 6162.4375, 6254.6259765625, 6346.814453125, 6439.0029296875, 6531.19140625, 6623.3798828125, 6715.56787109375, 6807.75634765625, 6899.9443359375, 6992.1328125, 7084.3212890625, 7176.509765625, 7268.6982421875, 7360.88671875, 7453.0751953125, 7545.26318359375, 7637.45166015625, 7729.6396484375, 7821.828125, 7914.0166015625, 8006.205078125, 8098.3935546875, 8190.58154296875, 8282.76953125, 8374.958984375, 8467.146484375, 8559.3359375, 8651.5234375, 8743.7119140625, 8835.900390625, 8928.0888671875, 9020.27734375, 9112.46484375, 9204.654296875, 9296.841796875, 9389.0302734375, 9481.21875, 9573.4072265625, 9665.595703125, 9757.783203125, 9849.97265625, 9942.16015625, 10034.349609375, 10126.537109375, 10218.7255859375, 10310.9140625, 10403.1025390625, 10495.291015625, 10587.478515625, 10679.66796875, 10771.85546875, 10864.044921875, 10956.232421875, 11048.4208984375, 11140.609375, 11232.7978515625, 11324.986328125, 11417.173828125, 11509.36328125, 11601.55078125, 11693.740234375, 11785.927734375, 11878.1162109375, 11970.3046875, 12062.4931640625, 12154.681640625, 12246.8701171875, 12339.0576171875, 12431.24609375, 12523.4345703125, 12615.623046875, 12707.8115234375, 12800.0]}, "name": "shift_sns", "inbound_nodes": [[["classifier3b/rs", 0, 0, {}]], [["classifier3b/rs", 0, 0, {}]], [["classifier3b/rs", 0, 0, {}]]]}, {"class_name": "Subtract", "config": {"name": "subtract", "trainable": true, "dtype": "float32"}, "name": "subtract", "inbound_nodes": [[["spectrum", 0, 0, {}], ["shift_sns", 0, 0, {}]], [["spectrum", 1, 0, {}], ["shift_sns", 1, 0, {}]], [["spectrum", 2, 0, {}], ["shift_sns", 2, 0, {}]]]}, {"class_name": "ReLU_", "config": {"name": "relu_residual", "trainable": true, "dtype": "float32"}, "name": "relu_residual", "inbound_nodes": [[["subtract", 0, 0, {}]], [["subtract", 1, 0, {}]], [["subtract", 2, 0, {}]]]}, {"class_name": "WeightedSum_", "config": {"name": "weighted_sum", "trainable": true, "dtype": "float32"}, "name": "weighted_sum", "inbound_nodes": [[["relu_residual", 0, 0, {}]], [["relu_residual", 1, 0, {}]], [["relu_residual", 2, 0, {}]]]}, {"class_name": "SFunction_", "config": {"name": "s_function", "trainable": true, "dtype": "float32", "alpha": 0.02, "beta": 2}, "name": "s_function", "inbound_nodes": [[["weighted_sum", 0, 0, {}]], [["weighted_sum", 1, 0, {}]], [["weighted_sum", 2, 0, {}]]]}], "input_layers": [["classifier3b/bm1", 0, 0], ["classifier3b/bm2", 0, 0], ["classifier3b/bm3", 0, 0], ["classifier3b/rs", 0, 0]], "output_layers": [["s_function", 0, 0], ["s_function", 1, 0], ["s_function", 2, 0]]}, "name": "classifier3b", "inbound_nodes": [[["split", 0, 0, {}], ["split", 0, 1, {}], ["split", 0, 2, {}], ["input_rs", 0, 0, {}]]], "shared_object_id": 14}, {"class_name": "Dummy_", "config": {"name": "classifier3b_dummy", "trainable": true, "dtype": "float32"}, "name": "classifier3b_dummy", "inbound_nodes": [[["classifier3b", 1, 0, {}], ["classifier3b", 1, 1, {}], ["classifier3b", 1, 2, {}]]], "shared_object_id": 15}, {"class_name": "AnyOf3_", "config": {"name": "any_of_3", "trainable": true, "dtype": "float32"}, "name": "any_of_3", "inbound_nodes": [[["classifier3b_dummy", 0, 0, {}], ["classifier3b_dummy", 0, 1, {}], ["classifier3b_dummy", 0, 2, {}]]], "shared_object_id": 16}], "input_layers": [["input_sp", 0, 0], ["input_rs", 0, 0]], "output_layers": [["any_of_3", 0, 0]]}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "sum_over_batch_size", "name": "bce_loss", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 20}, "metrics": [[{"class_name": "BinaryAccuracy", "config": {"name": "accuracy", "dtype": "float32", "threshold": 0.3}, "shared_object_id": 21}, {"class_name": "Recall", "config": {"name": "recall", "dtype": "float32", "thresholds": 0.3, "top_k": null, "class_id": null}, "shared_object_id": 22}, {"class_name": "Precision", "config": {"name": "precision", "dtype": "float32", "thresholds": 0.3, "top_k": null, "class_id": null}, "shared_object_id": 23}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 0.0005, "decay": 0.004, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07}}}}

#_self_saveable_object_factories"ņ
_tf_keras_input_layerŅ{"class_name": "InputLayer", "name": "input_sp", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 96]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 96]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_sp"}}
Ļ
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
“__call__
+µ&call_and_return_all_conditional_losses"
_tf_keras_layer’{"name": "insert_channel_axis", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "InsertChannelAxis_", "config": {"name": "insert_channel_axis", "trainable": true, "dtype": "float32", "ch_axis": -1}, "inbound_nodes": [[["input_sp", 0, 0, {}]]], "shared_object_id": 1}
¾
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
¶__call__
+·&call_and_return_all_conditional_losses"
_tf_keras_layerī{"name": "split", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Split_", "config": {"name": "split", "trainable": true, "dtype": "float32", "axis": -3, "n_splits": 3}, "inbound_nodes": [[["insert_channel_axis", 0, 0, {}]]], "shared_object_id": 2}

#_self_saveable_object_factories"ā
_tf_keras_input_layerĀ{"class_name": "InputLayer", "name": "input_rs", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_rs"}}
Ōx
layer-0
layer-1
layer-2
layer-3
layer-4
 layer_with_weights-0
 layer-5
!layer-6
"layer-7
#layer_with_weights-1
#layer-8
$layer_with_weights-2
$layer-9
#%_self_saveable_object_factories
&	variables
'regularization_losses
(trainable_variables
)	keras_api
ø__call__
+¹&call_and_return_all_conditional_losses"Īu
_tf_keras_network²u{"name": "classifier3b", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "classifier3b", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "classifier3b/bm1"}, "name": "classifier3b/bm1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "classifier3b/rs"}, "name": "classifier3b/rs", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "classifier3b/bm2"}, "name": "classifier3b/bm2", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "classifier3b/bm3"}, "name": "classifier3b/bm3", "inbound_nodes": []}, {"class_name": "Spectrum_", "config": {"name": "spectrum", "trainable": true, "dtype": "float32", "t_axis": -2, "ch_axis": -1}, "name": "spectrum", "inbound_nodes": [[["classifier3b/bm1", 0, 0, {}]], [["classifier3b/bm2", 0, 0, {}]], [["classifier3b/bm3", 0, 0, {}]]]}, {"class_name": "ShiftSNS_", "config": {"name": "shift_sns", "trainable": true, "dtype": "float32", "sns_shape": [96], "cutin_freq": 4000, "isns": [-51.73952102661133, -51.91908264160156, -52.17738723754883, -52.491268157958984, -52.74644470214844, -53.01625061035156, -53.32057571411133, -53.5711784362793, -53.81317138671875, -54.05160903930664, -54.34244918823242, -54.645511627197266, -54.94461441040039, -55.16508483886719, -55.3690185546875, -55.55717468261719, -55.71791076660156, -55.86759948730469, -56.01735305786133, -56.17418670654297, -56.310523986816406, -56.43460464477539, -56.54563903808594, -56.64398956298828, -56.74330520629883, -56.84583282470703, -56.98292922973633, -57.1292724609375, -57.27627182006836, -57.38688659667969, -57.48052978515625, -57.5677375793457, -57.65999984741211, -57.707054138183594, -57.74354934692383, -57.785884857177734, -57.789222717285156, -57.77459716796875, -57.74294662475586, -57.74364471435547, -57.736480712890625, -57.722232818603516, -57.75682830810547, -57.78118133544922, -57.803714752197266, -57.85961151123047, -57.92207717895508, -57.99568176269531, -58.09674072265625, -58.18413543701172, -58.263145446777344, -58.332786560058594, -58.412906646728516, -58.507652282714844, -58.621131896972656, -58.73277282714844, -58.814598083496094, -58.86747360229492, -58.936519622802734, -58.98716735839844, -59.031463623046875, -59.116207122802734, -59.16674041748047, -59.193458557128906, -59.19350051879883, -59.217323303222656, -59.24707794189453, -59.274566650390625, -59.22399139404297, -59.18412780761719, -59.17753982543945, -59.254913330078125, -59.29663848876953, -59.29777145385742, -59.27923583984375, -59.28166580200195, -59.29579162597656, -59.292144775390625, -59.28715515136719, -59.2860107421875, -59.300048828125, -59.321205139160156, -59.35308837890625, -59.403472900390625, -59.459136962890625, -59.5334587097168, -59.63352966308594, -59.789154052734375, -60.00239181518555, -60.27350616455078, -60.605594635009766, -60.99026870727539, -61.40313720703125, -61.763999938964844, -62.23042297363281, -62.628910064697266], "F": [4042.10546875, 4134.2939453125, 4226.482421875, 4318.67041015625, 4410.85888671875, 4503.04736328125, 4595.23583984375, 4687.423828125, 4779.6123046875, 4871.80078125, 4963.9892578125, 5056.17724609375, 5148.36572265625, 5240.55419921875, 5332.74267578125, 5424.9306640625, 5517.119140625, 5609.3076171875, 5701.49609375, 5793.6845703125, 5885.87255859375, 5978.06103515625, 6070.24951171875, 6162.4375, 6254.6259765625, 6346.814453125, 6439.0029296875, 6531.19140625, 6623.3798828125, 6715.56787109375, 6807.75634765625, 6899.9443359375, 6992.1328125, 7084.3212890625, 7176.509765625, 7268.6982421875, 7360.88671875, 7453.0751953125, 7545.26318359375, 7637.45166015625, 7729.6396484375, 7821.828125, 7914.0166015625, 8006.205078125, 8098.3935546875, 8190.58154296875, 8282.76953125, 8374.958984375, 8467.146484375, 8559.3359375, 8651.5234375, 8743.7119140625, 8835.900390625, 8928.0888671875, 9020.27734375, 9112.46484375, 9204.654296875, 9296.841796875, 9389.0302734375, 9481.21875, 9573.4072265625, 9665.595703125, 9757.783203125, 9849.97265625, 9942.16015625, 10034.349609375, 10126.537109375, 10218.7255859375, 10310.9140625, 10403.1025390625, 10495.291015625, 10587.478515625, 10679.66796875, 10771.85546875, 10864.044921875, 10956.232421875, 11048.4208984375, 11140.609375, 11232.7978515625, 11324.986328125, 11417.173828125, 11509.36328125, 11601.55078125, 11693.740234375, 11785.927734375, 11878.1162109375, 11970.3046875, 12062.4931640625, 12154.681640625, 12246.8701171875, 12339.0576171875, 12431.24609375, 12523.4345703125, 12615.623046875, 12707.8115234375, 12800.0]}, "name": "shift_sns", "inbound_nodes": [[["classifier3b/rs", 0, 0, {}]], [["classifier3b/rs", 0, 0, {}]], [["classifier3b/rs", 0, 0, {}]]]}, {"class_name": "Subtract", "config": {"name": "subtract", "trainable": true, "dtype": "float32"}, "name": "subtract", "inbound_nodes": [[["spectrum", 0, 0, {}], ["shift_sns", 0, 0, {}]], [["spectrum", 1, 0, {}], ["shift_sns", 1, 0, {}]], [["spectrum", 2, 0, {}], ["shift_sns", 2, 0, {}]]]}, {"class_name": "ReLU_", "config": {"name": "relu_residual", "trainable": true, "dtype": "float32"}, "name": "relu_residual", "inbound_nodes": [[["subtract", 0, 0, {}]], [["subtract", 1, 0, {}]], [["subtract", 2, 0, {}]]]}, {"class_name": "WeightedSum_", "config": {"name": "weighted_sum", "trainable": true, "dtype": "float32"}, "name": "weighted_sum", "inbound_nodes": [[["relu_residual", 0, 0, {}]], [["relu_residual", 1, 0, {}]], [["relu_residual", 2, 0, {}]]]}, {"class_name": "SFunction_", "config": {"name": "s_function", "trainable": true, "dtype": "float32", "alpha": 0.02, "beta": 2}, "name": "s_function", "inbound_nodes": [[["weighted_sum", 0, 0, {}]], [["weighted_sum", 1, 0, {}]], [["weighted_sum", 2, 0, {}]]]}], "input_layers": [["classifier3b/bm1", 0, 0], ["classifier3b/bm2", 0, 0], ["classifier3b/bm3", 0, 0], ["classifier3b/rs", 0, 0]], "output_layers": [["s_function", 0, 0], ["s_function", 1, 0], ["s_function", 2, 0]]}, "inbound_nodes": [[["split", 0, 0, {}], ["split", 0, 1, {}], ["split", 0, 2, {}], ["input_rs", 0, 0, {}]]], "shared_object_id": 14, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 96, 32, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 96, 32, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 96, 32, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null]}, "ndim": 1, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 96, 32, 1]}, {"class_name": "TensorShape", "items": [null, 96, 32, 1]}, {"class_name": "TensorShape", "items": [null, 96, 32, 1]}, {"class_name": "TensorShape", "items": [null]}], "is_graph_network": true, "save_spec": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 96, 32, 1]}, "float32", "classifier3b/bm1"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 96, 32, 1]}, "float32", "classifier3b/bm2"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 96, 32, 1]}, "float32", "classifier3b/bm3"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null]}, "float32", "classifier3b/rs"]}], "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "classifier3b", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "classifier3b/bm1"}, "name": "classifier3b/bm1", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "classifier3b/rs"}, "name": "classifier3b/rs", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "classifier3b/bm2"}, "name": "classifier3b/bm2", "inbound_nodes": [], "shared_object_id": 6}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "classifier3b/bm3"}, "name": "classifier3b/bm3", "inbound_nodes": [], "shared_object_id": 7}, {"class_name": "Spectrum_", "config": {"name": "spectrum", "trainable": true, "dtype": "float32", "t_axis": -2, "ch_axis": -1}, "name": "spectrum", "inbound_nodes": [[["classifier3b/bm1", 0, 0, {}]], [["classifier3b/bm2", 0, 0, {}]], [["classifier3b/bm3", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "ShiftSNS_", "config": {"name": "shift_sns", "trainable": true, "dtype": "float32", "sns_shape": [96], "cutin_freq": 4000, "isns": [-51.73952102661133, -51.91908264160156, -52.17738723754883, -52.491268157958984, -52.74644470214844, -53.01625061035156, -53.32057571411133, -53.5711784362793, -53.81317138671875, -54.05160903930664, -54.34244918823242, -54.645511627197266, -54.94461441040039, -55.16508483886719, -55.3690185546875, -55.55717468261719, -55.71791076660156, -55.86759948730469, -56.01735305786133, -56.17418670654297, -56.310523986816406, -56.43460464477539, -56.54563903808594, -56.64398956298828, -56.74330520629883, -56.84583282470703, -56.98292922973633, -57.1292724609375, -57.27627182006836, -57.38688659667969, -57.48052978515625, -57.5677375793457, -57.65999984741211, -57.707054138183594, -57.74354934692383, -57.785884857177734, -57.789222717285156, -57.77459716796875, -57.74294662475586, -57.74364471435547, -57.736480712890625, -57.722232818603516, -57.75682830810547, -57.78118133544922, -57.803714752197266, -57.85961151123047, -57.92207717895508, -57.99568176269531, -58.09674072265625, -58.18413543701172, -58.263145446777344, -58.332786560058594, -58.412906646728516, -58.507652282714844, -58.621131896972656, -58.73277282714844, -58.814598083496094, -58.86747360229492, -58.936519622802734, -58.98716735839844, -59.031463623046875, -59.116207122802734, -59.16674041748047, -59.193458557128906, -59.19350051879883, -59.217323303222656, -59.24707794189453, -59.274566650390625, -59.22399139404297, -59.18412780761719, -59.17753982543945, -59.254913330078125, -59.29663848876953, -59.29777145385742, -59.27923583984375, -59.28166580200195, -59.29579162597656, -59.292144775390625, -59.28715515136719, -59.2860107421875, -59.300048828125, -59.321205139160156, -59.35308837890625, -59.403472900390625, -59.459136962890625, -59.5334587097168, -59.63352966308594, -59.789154052734375, -60.00239181518555, -60.27350616455078, -60.605594635009766, -60.99026870727539, -61.40313720703125, -61.763999938964844, -62.23042297363281, -62.628910064697266], "F": [4042.10546875, 4134.2939453125, 4226.482421875, 4318.67041015625, 4410.85888671875, 4503.04736328125, 4595.23583984375, 4687.423828125, 4779.6123046875, 4871.80078125, 4963.9892578125, 5056.17724609375, 5148.36572265625, 5240.55419921875, 5332.74267578125, 5424.9306640625, 5517.119140625, 5609.3076171875, 5701.49609375, 5793.6845703125, 5885.87255859375, 5978.06103515625, 6070.24951171875, 6162.4375, 6254.6259765625, 6346.814453125, 6439.0029296875, 6531.19140625, 6623.3798828125, 6715.56787109375, 6807.75634765625, 6899.9443359375, 6992.1328125, 7084.3212890625, 7176.509765625, 7268.6982421875, 7360.88671875, 7453.0751953125, 7545.26318359375, 7637.45166015625, 7729.6396484375, 7821.828125, 7914.0166015625, 8006.205078125, 8098.3935546875, 8190.58154296875, 8282.76953125, 8374.958984375, 8467.146484375, 8559.3359375, 8651.5234375, 8743.7119140625, 8835.900390625, 8928.0888671875, 9020.27734375, 9112.46484375, 9204.654296875, 9296.841796875, 9389.0302734375, 9481.21875, 9573.4072265625, 9665.595703125, 9757.783203125, 9849.97265625, 9942.16015625, 10034.349609375, 10126.537109375, 10218.7255859375, 10310.9140625, 10403.1025390625, 10495.291015625, 10587.478515625, 10679.66796875, 10771.85546875, 10864.044921875, 10956.232421875, 11048.4208984375, 11140.609375, 11232.7978515625, 11324.986328125, 11417.173828125, 11509.36328125, 11601.55078125, 11693.740234375, 11785.927734375, 11878.1162109375, 11970.3046875, 12062.4931640625, 12154.681640625, 12246.8701171875, 12339.0576171875, 12431.24609375, 12523.4345703125, 12615.623046875, 12707.8115234375, 12800.0]}, "name": "shift_sns", "inbound_nodes": [[["classifier3b/rs", 0, 0, {}]], [["classifier3b/rs", 0, 0, {}]], [["classifier3b/rs", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Subtract", "config": {"name": "subtract", "trainable": true, "dtype": "float32"}, "name": "subtract", "inbound_nodes": [[["spectrum", 0, 0, {}], ["shift_sns", 0, 0, {}]], [["spectrum", 1, 0, {}], ["shift_sns", 1, 0, {}]], [["spectrum", 2, 0, {}], ["shift_sns", 2, 0, {}]]], "shared_object_id": 10}, {"class_name": "ReLU_", "config": {"name": "relu_residual", "trainable": true, "dtype": "float32"}, "name": "relu_residual", "inbound_nodes": [[["subtract", 0, 0, {}]], [["subtract", 1, 0, {}]], [["subtract", 2, 0, {}]]], "shared_object_id": 11}, {"class_name": "WeightedSum_", "config": {"name": "weighted_sum", "trainable": true, "dtype": "float32"}, "name": "weighted_sum", "inbound_nodes": [[["relu_residual", 0, 0, {}]], [["relu_residual", 1, 0, {}]], [["relu_residual", 2, 0, {}]]], "shared_object_id": 12}, {"class_name": "SFunction_", "config": {"name": "s_function", "trainable": true, "dtype": "float32", "alpha": 0.02, "beta": 2}, "name": "s_function", "inbound_nodes": [[["weighted_sum", 0, 0, {}]], [["weighted_sum", 1, 0, {}]], [["weighted_sum", 2, 0, {}]]], "shared_object_id": 13}], "input_layers": [["classifier3b/bm1", 0, 0], ["classifier3b/bm2", 0, 0], ["classifier3b/bm3", 0, 0], ["classifier3b/rs", 0, 0]], "output_layers": [["s_function", 0, 0], ["s_function", 1, 0], ["s_function", 2, 0]]}}}
ļ
#*_self_saveable_object_factories
+	variables
,regularization_losses
-trainable_variables
.	keras_api
ŗ__call__
+»&call_and_return_all_conditional_losses"¹
_tf_keras_layer{"name": "classifier3b_dummy", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dummy_", "config": {"name": "classifier3b_dummy", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["classifier3b", 1, 0, {}], ["classifier3b", 1, 1, {}], ["classifier3b", 1, 2, {}]]], "shared_object_id": 15}
ī
#/_self_saveable_object_factories
0	variables
1regularization_losses
2trainable_variables
3	keras_api
¼__call__
+½&call_and_return_all_conditional_losses"ø
_tf_keras_layer{"name": "any_of_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "AnyOf3_", "config": {"name": "any_of_3", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["classifier3b_dummy", 0, 0, {}], ["classifier3b_dummy", 0, 1, {}], ["classifier3b_dummy", 0, 2, {}]]], "shared_object_id": 16}
"
	optimizer
 "
trackable_dict_wrapper
<
40
51
62
73"
trackable_list_wrapper
 "
trackable_list_wrapper
<
40
51
62
73"
trackable_list_wrapper
Ī

	variables
regularization_losses
8metrics
trainable_variables
9layer_metrics

:layers
;non_trainable_variables
<layer_regularization_losses
²__call__
±_default_save_signature
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
-
¾serving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
	variables
regularization_losses
=metrics
trainable_variables
>layer_metrics

?layers
@non_trainable_variables
Alayer_regularization_losses
“__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
	variables
regularization_losses
Bmetrics
trainable_variables
Clayer_metrics

Dlayers
Enon_trainable_variables
Flayer_regularization_losses
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
°
#G_self_saveable_object_factories"
_tf_keras_input_layerč{"class_name": "InputLayer", "name": "classifier3b/bm1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 32, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "classifier3b/bm1"}}

#H_self_saveable_object_factories"š
_tf_keras_input_layerŠ{"class_name": "InputLayer", "name": "classifier3b/rs", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "classifier3b/rs"}}
°
#I_self_saveable_object_factories"
_tf_keras_input_layerč{"class_name": "InputLayer", "name": "classifier3b/bm2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 32, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "classifier3b/bm2"}}
°
#J_self_saveable_object_factories"
_tf_keras_input_layerč{"class_name": "InputLayer", "name": "classifier3b/bm3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 32, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "classifier3b/bm3"}}

Kremove_axes
#L_self_saveable_object_factories
M	variables
Nregularization_losses
Otrainable_variables
P	keras_api
æ__call__
+Ą&call_and_return_all_conditional_losses"Ō
_tf_keras_layerŗ{"name": "spectrum", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Spectrum_", "config": {"name": "spectrum", "trainable": true, "dtype": "float32", "t_axis": -2, "ch_axis": -1}, "inbound_nodes": [[["classifier3b/bm1", 0, 0, {}]], [["classifier3b/bm2", 0, 0, {}]], [["classifier3b/bm3", 0, 0, {}]]], "shared_object_id": 8}
!
Q	sns_shape
4sns
Rexpand_axes
#S_self_saveable_object_factories
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
Į__call__
+Ā&call_and_return_all_conditional_losses"±
_tf_keras_layer{"name": "shift_sns", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ShiftSNS_", "config": {"name": "shift_sns", "trainable": true, "dtype": "float32", "sns_shape": [96], "cutin_freq": 4000, "isns": [-51.73952102661133, -51.91908264160156, -52.17738723754883, -52.491268157958984, -52.74644470214844, -53.01625061035156, -53.32057571411133, -53.5711784362793, -53.81317138671875, -54.05160903930664, -54.34244918823242, -54.645511627197266, -54.94461441040039, -55.16508483886719, -55.3690185546875, -55.55717468261719, -55.71791076660156, -55.86759948730469, -56.01735305786133, -56.17418670654297, -56.310523986816406, -56.43460464477539, -56.54563903808594, -56.64398956298828, -56.74330520629883, -56.84583282470703, -56.98292922973633, -57.1292724609375, -57.27627182006836, -57.38688659667969, -57.48052978515625, -57.5677375793457, -57.65999984741211, -57.707054138183594, -57.74354934692383, -57.785884857177734, -57.789222717285156, -57.77459716796875, -57.74294662475586, -57.74364471435547, -57.736480712890625, -57.722232818603516, -57.75682830810547, -57.78118133544922, -57.803714752197266, -57.85961151123047, -57.92207717895508, -57.99568176269531, -58.09674072265625, -58.18413543701172, -58.263145446777344, -58.332786560058594, -58.412906646728516, -58.507652282714844, -58.621131896972656, -58.73277282714844, -58.814598083496094, -58.86747360229492, -58.936519622802734, -58.98716735839844, -59.031463623046875, -59.116207122802734, -59.16674041748047, -59.193458557128906, -59.19350051879883, -59.217323303222656, -59.24707794189453, -59.274566650390625, -59.22399139404297, -59.18412780761719, -59.17753982543945, -59.254913330078125, -59.29663848876953, -59.29777145385742, -59.27923583984375, -59.28166580200195, -59.29579162597656, -59.292144775390625, -59.28715515136719, -59.2860107421875, -59.300048828125, -59.321205139160156, -59.35308837890625, -59.403472900390625, -59.459136962890625, -59.5334587097168, -59.63352966308594, -59.789154052734375, -60.00239181518555, -60.27350616455078, -60.605594635009766, -60.99026870727539, -61.40313720703125, -61.763999938964844, -62.23042297363281, -62.628910064697266], "F": [4042.10546875, 4134.2939453125, 4226.482421875, 4318.67041015625, 4410.85888671875, 4503.04736328125, 4595.23583984375, 4687.423828125, 4779.6123046875, 4871.80078125, 4963.9892578125, 5056.17724609375, 5148.36572265625, 5240.55419921875, 5332.74267578125, 5424.9306640625, 5517.119140625, 5609.3076171875, 5701.49609375, 5793.6845703125, 5885.87255859375, 5978.06103515625, 6070.24951171875, 6162.4375, 6254.6259765625, 6346.814453125, 6439.0029296875, 6531.19140625, 6623.3798828125, 6715.56787109375, 6807.75634765625, 6899.9443359375, 6992.1328125, 7084.3212890625, 7176.509765625, 7268.6982421875, 7360.88671875, 7453.0751953125, 7545.26318359375, 7637.45166015625, 7729.6396484375, 7821.828125, 7914.0166015625, 8006.205078125, 8098.3935546875, 8190.58154296875, 8282.76953125, 8374.958984375, 8467.146484375, 8559.3359375, 8651.5234375, 8743.7119140625, 8835.900390625, 8928.0888671875, 9020.27734375, 9112.46484375, 9204.654296875, 9296.841796875, 9389.0302734375, 9481.21875, 9573.4072265625, 9665.595703125, 9757.783203125, 9849.97265625, 9942.16015625, 10034.349609375, 10126.537109375, 10218.7255859375, 10310.9140625, 10403.1025390625, 10495.291015625, 10587.478515625, 10679.66796875, 10771.85546875, 10864.044921875, 10956.232421875, 11048.4208984375, 11140.609375, 11232.7978515625, 11324.986328125, 11417.173828125, 11509.36328125, 11601.55078125, 11693.740234375, 11785.927734375, 11878.1162109375, 11970.3046875, 12062.4931640625, 12154.681640625, 12246.8701171875, 12339.0576171875, 12431.24609375, 12523.4345703125, 12615.623046875, 12707.8115234375, 12800.0]}, "inbound_nodes": [[["classifier3b/rs", 0, 0, {}]], [["classifier3b/rs", 0, 0, {}]], [["classifier3b/rs", 0, 0, {}]]], "shared_object_id": 9}

#X_self_saveable_object_factories
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
Ć__call__
+Ä&call_and_return_all_conditional_losses"é
_tf_keras_layerĻ{"name": "subtract", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Subtract", "config": {"name": "subtract", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["spectrum", 0, 0, {}], ["shift_sns", 0, 0, {}]], [["spectrum", 1, 0, {}], ["shift_sns", 1, 0, {}]], [["spectrum", 2, 0, {}], ["shift_sns", 2, 0, {}]]], "shared_object_id": 10, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 96]}, {"class_name": "TensorShape", "items": [null, 96]}]}
Ü
#]_self_saveable_object_factories
^	variables
_regularization_losses
`trainable_variables
a	keras_api
Å__call__
+Ę&call_and_return_all_conditional_losses"¦
_tf_keras_layer{"name": "relu_residual", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ReLU_", "config": {"name": "relu_residual", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["subtract", 0, 0, {}]], [["subtract", 1, 0, {}]], [["subtract", 2, 0, {}]]], "shared_object_id": 11}

5w
baxes
#c_self_saveable_object_factories
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
Ē__call__
+Č&call_and_return_all_conditional_losses"ŗ
_tf_keras_layer {"name": "weighted_sum", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "WeightedSum_", "config": {"name": "weighted_sum", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["relu_residual", 0, 0, {}]], [["relu_residual", 1, 0, {}]], [["relu_residual", 2, 0, {}]]], "shared_object_id": 12}

	6alpha
7beta
#h_self_saveable_object_factories
i	variables
jregularization_losses
ktrainable_variables
l	keras_api
É__call__
+Ź&call_and_return_all_conditional_losses"Ė
_tf_keras_layer±{"name": "s_function", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "SFunction_", "config": {"name": "s_function", "trainable": true, "dtype": "float32", "alpha": 0.02, "beta": 2}, "inbound_nodes": [[["weighted_sum", 0, 0, {}]], [["weighted_sum", 1, 0, {}]], [["weighted_sum", 2, 0, {}]]], "shared_object_id": 13}
 "
trackable_dict_wrapper
<
40
51
62
73"
trackable_list_wrapper
 "
trackable_list_wrapper
<
40
51
62
73"
trackable_list_wrapper
°
&	variables
'regularization_losses
mmetrics
(trainable_variables
nlayer_metrics

olayers
pnon_trainable_variables
qlayer_regularization_losses
ø__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
+	variables
,regularization_losses
rmetrics
-trainable_variables
slayer_metrics

tlayers
unon_trainable_variables
vlayer_regularization_losses
ŗ__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
0	variables
1regularization_losses
wmetrics
2trainable_variables
xlayer_metrics

ylayers
znon_trainable_variables
{layer_regularization_losses
¼__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
:`2shift_sns/sns
": `2weighted_sum/weights
: 2s_function/alpha
: 2s_function/beta
<
|0
}1
~2
3"
trackable_list_wrapper
 "
trackable_dict_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
M	variables
Nregularization_losses
metrics
Otrainable_variables
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
æ__call__
+Ą&call_and_return_all_conditional_losses
'Ą"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
40"
trackable_list_wrapper
 "
trackable_list_wrapper
'
40"
trackable_list_wrapper
µ
T	variables
Uregularization_losses
metrics
Vtrainable_variables
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
Į__call__
+Ā&call_and_return_all_conditional_losses
'Ā"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Y	variables
Zregularization_losses
metrics
[trainable_variables
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
Ć__call__
+Ä&call_and_return_all_conditional_losses
'Ä"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
^	variables
_regularization_losses
metrics
`trainable_variables
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
Å__call__
+Ę&call_and_return_all_conditional_losses
'Ę"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
50"
trackable_list_wrapper
 "
trackable_list_wrapper
'
50"
trackable_list_wrapper
µ
d	variables
eregularization_losses
metrics
ftrainable_variables
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
Ē__call__
+Č&call_and_return_all_conditional_losses
'Č"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
µ
i	variables
jregularization_losses
metrics
ktrainable_variables
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
É__call__
+Ź&call_and_return_all_conditional_losses
'Ź"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
f
0
1
2
3
4
 5
!6
"7
#8
$9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ų

total

count
 	variables
”	keras_api"
_tf_keras_metric{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 28}


¢total

£count
¤
_fn_kwargs
„	variables
¦	keras_api"Į
_tf_keras_metric¦{"class_name": "BinaryAccuracy", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "threshold": 0.3}, "shared_object_id": 21}
¶
§
thresholds
Øtrue_positives
©false_negatives
Ŗ	variables
«	keras_api"×
_tf_keras_metric¼{"class_name": "Recall", "name": "recall", "dtype": "float32", "config": {"name": "recall", "dtype": "float32", "thresholds": 0.3, "top_k": null, "class_id": null}, "shared_object_id": 22}
æ
¬
thresholds
­true_positives
®false_positives
Æ	variables
°	keras_api"ą
_tf_keras_metricÅ{"class_name": "Precision", "name": "precision", "dtype": "float32", "config": {"name": "precision", "dtype": "float32", "thresholds": 0.3, "top_k": null, "class_id": null}, "shared_object_id": 23}
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
 	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
¢0
£1"
trackable_list_wrapper
.
„	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
Ø0
©1"
trackable_list_wrapper
.
Ŗ	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
­0
®1"
trackable_list_wrapper
.
Æ	variables"
_generic_user_object
2
 __inference__wrapped_model_41372ą
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *P¢M
KH
&#
input_sp’’’’’’’’’``

input_rs’’’’’’’’’
ö2ó
*__inference_physicalnn_layer_call_fn_41680
*__inference_physicalnn_layer_call_fn_41824
*__inference_physicalnn_layer_call_fn_41838
*__inference_physicalnn_layer_call_fn_41746Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ā2ß
E__inference_physicalnn_layer_call_and_return_conditional_losses_41885
E__inference_physicalnn_layer_call_and_return_conditional_losses_41932
E__inference_physicalnn_layer_call_and_return_conditional_losses_41770
E__inference_physicalnn_layer_call_and_return_conditional_losses_41794Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ó2Š
3__inference_insert_channel_axis_layer_call_fn_40537
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ī2ė
N__inference_insert_channel_axis_layer_call_and_return_conditional_losses_39219
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Å2Ā
%__inference_split_layer_call_fn_39572
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ą2Ż
@__inference_split_layer_call_and_return_conditional_losses_39597
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ž2ū
,__inference_classifier3b_layer_call_fn_41444
,__inference_classifier3b_layer_call_fn_41952
,__inference_classifier3b_layer_call_fn_41972
,__inference_classifier3b_layer_call_fn_41554Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ź2ē
G__inference_classifier3b_layer_call_and_return_conditional_losses_42015
G__inference_classifier3b_layer_call_and_return_conditional_losses_42058
G__inference_classifier3b_layer_call_and_return_conditional_losses_41597
G__inference_classifier3b_layer_call_and_return_conditional_losses_41640Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ņ2Ļ
2__inference_classifier3b_dummy_layer_call_fn_39552
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ķ2ź
M__inference_classifier3b_dummy_layer_call_and_return_conditional_losses_39586
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Č2Å
(__inference_any_of_3_layer_call_fn_38345
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ć2ą
C__inference_any_of_3_layer_call_and_return_conditional_losses_40766
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ÓBŠ
#__inference_signature_wrapper_41810input_rsinput_sp"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Č2Å
(__inference_spectrum_layer_call_fn_38645
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ć2ą
C__inference_spectrum_layer_call_and_return_conditional_losses_38317
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
É2Ę
)__inference_shift_sns_layer_call_fn_40217
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ä2į
D__inference_shift_sns_layer_call_and_return_conditional_losses_39056
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ņ2Ļ
(__inference_subtract_layer_call_fn_42064¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ķ2ź
C__inference_subtract_layer_call_and_return_conditional_losses_42070¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ķ2Ź
-__inference_relu_residual_layer_call_fn_38355
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
č2å
H__inference_relu_residual_layer_call_and_return_conditional_losses_39061
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ģ2É
,__inference_weighted_sum_layer_call_fn_39081
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ē2ä
G__inference_weighted_sum_layer_call_and_return_conditional_losses_38725
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ź2Ē
*__inference_s_function_layer_call_fn_38711
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
å2ā
E__inference_s_function_layer_call_and_return_conditional_losses_38736
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ø
 __inference__wrapped_model_413724576Z¢W
P¢M
KH
&#
input_sp’’’’’’’’’``

input_rs’’’’’’’’’
Ŗ "/Ŗ,
*
any_of_3
any_of_3’’’’’’’’’ß
C__inference_any_of_3_layer_call_and_return_conditional_losses_40766r¢o
h¢e
c`

inputs/0’’’’’’’’’

inputs/1’’’’’’’’’

inputs/2’’’’’’’’’
Ŗ "!¢

0’’’’’’’’’
 ·
(__inference_any_of_3_layer_call_fn_38345r¢o
h¢e
c`

inputs/0’’’’’’’’’

inputs/1’’’’’’’’’

inputs/2’’’’’’’’’
Ŗ "’’’’’’’’’¦
M__inference_classifier3b_dummy_layer_call_and_return_conditional_losses_39586Ōr¢o
h¢e
c`

inputs/0’’’’’’’’’

inputs/1’’’’’’’’’

inputs/2’’’’’’’’’
Ŗ "^¢[
TQ

0/0’’’’’’’’’

0/1’’’’’’’’’

0/2’’’’’’’’’
 ū
2__inference_classifier3b_dummy_layer_call_fn_39552Är¢o
h¢e
c`

inputs/0’’’’’’’’’

inputs/1’’’’’’’’’

inputs/2’’’’’’’’’
Ŗ "NK

0’’’’’’’’’

1’’’’’’’’’

2’’’’’’’’’
G__inference_classifier3b_layer_call_and_return_conditional_losses_41597Ė4576ā¢Ž
Ö¢Ņ
ĒĆ
2/
classifier3b/bm1’’’’’’’’’` 
2/
classifier3b/bm2’’’’’’’’’` 
2/
classifier3b/bm3’’’’’’’’’` 
%"
classifier3b/rs’’’’’’’’’
p 

 
Ŗ "^¢[
TQ

0/0’’’’’’’’’

0/1’’’’’’’’’

0/2’’’’’’’’’
 
G__inference_classifier3b_layer_call_and_return_conditional_losses_41640Ė4576ā¢Ž
Ö¢Ņ
ĒĆ
2/
classifier3b/bm1’’’’’’’’’` 
2/
classifier3b/bm2’’’’’’’’’` 
2/
classifier3b/bm3’’’’’’’’’` 
%"
classifier3b/rs’’’’’’’’’
p

 
Ŗ "^¢[
TQ

0/0’’’’’’’’’

0/1’’’’’’’’’

0/2’’’’’’’’’
 ų
G__inference_classifier3b_layer_call_and_return_conditional_losses_42015¬4576Ć¢æ
·¢³
Ø¤
*'
inputs/0’’’’’’’’’` 
*'
inputs/1’’’’’’’’’` 
*'
inputs/2’’’’’’’’’` 

inputs/3’’’’’’’’’
p 

 
Ŗ "^¢[
TQ

0/0’’’’’’’’’

0/1’’’’’’’’’

0/2’’’’’’’’’
 ų
G__inference_classifier3b_layer_call_and_return_conditional_losses_42058¬4576Ć¢æ
·¢³
Ø¤
*'
inputs/0’’’’’’’’’` 
*'
inputs/1’’’’’’’’’` 
*'
inputs/2’’’’’’’’’` 

inputs/3’’’’’’’’’
p

 
Ŗ "^¢[
TQ

0/0’’’’’’’’’

0/1’’’’’’’’’

0/2’’’’’’’’’
 ģ
,__inference_classifier3b_layer_call_fn_41444»4576ā¢Ž
Ö¢Ņ
ĒĆ
2/
classifier3b/bm1’’’’’’’’’` 
2/
classifier3b/bm2’’’’’’’’’` 
2/
classifier3b/bm3’’’’’’’’’` 
%"
classifier3b/rs’’’’’’’’’
p 

 
Ŗ "NK

0’’’’’’’’’

1’’’’’’’’’

2’’’’’’’’’ģ
,__inference_classifier3b_layer_call_fn_41554»4576ā¢Ž
Ö¢Ņ
ĒĆ
2/
classifier3b/bm1’’’’’’’’’` 
2/
classifier3b/bm2’’’’’’’’’` 
2/
classifier3b/bm3’’’’’’’’’` 
%"
classifier3b/rs’’’’’’’’’
p

 
Ŗ "NK

0’’’’’’’’’

1’’’’’’’’’

2’’’’’’’’’Ķ
,__inference_classifier3b_layer_call_fn_419524576Ć¢æ
·¢³
Ø¤
*'
inputs/0’’’’’’’’’` 
*'
inputs/1’’’’’’’’’` 
*'
inputs/2’’’’’’’’’` 

inputs/3’’’’’’’’’
p 

 
Ŗ "NK

0’’’’’’’’’

1’’’’’’’’’

2’’’’’’’’’Ķ
,__inference_classifier3b_layer_call_fn_419724576Ć¢æ
·¢³
Ø¤
*'
inputs/0’’’’’’’’’` 
*'
inputs/1’’’’’’’’’` 
*'
inputs/2’’’’’’’’’` 

inputs/3’’’’’’’’’
p

 
Ŗ "NK

0’’’’’’’’’

1’’’’’’’’’

2’’’’’’’’’¶
N__inference_insert_channel_axis_layer_call_and_return_conditional_losses_39219d3¢0
)¢&
$!
inputs’’’’’’’’’``
Ŗ "-¢*
# 
0’’’’’’’’’``
 
3__inference_insert_channel_axis_layer_call_fn_40537W3¢0
)¢&
$!
inputs’’’’’’’’’``
Ŗ " ’’’’’’’’’``×
E__inference_physicalnn_layer_call_and_return_conditional_losses_417704576b¢_
X¢U
KH
&#
input_sp’’’’’’’’’``

input_rs’’’’’’’’’
p 

 
Ŗ "!¢

0’’’’’’’’’
 ×
E__inference_physicalnn_layer_call_and_return_conditional_losses_417944576b¢_
X¢U
KH
&#
input_sp’’’’’’’’’``

input_rs’’’’’’’’’
p

 
Ŗ "!¢

0’’’’’’’’’
 ×
E__inference_physicalnn_layer_call_and_return_conditional_losses_418854576b¢_
X¢U
KH
&#
inputs/0’’’’’’’’’``

inputs/1’’’’’’’’’
p 

 
Ŗ "!¢

0’’’’’’’’’
 ×
E__inference_physicalnn_layer_call_and_return_conditional_losses_419324576b¢_
X¢U
KH
&#
inputs/0’’’’’’’’’``

inputs/1’’’’’’’’’
p

 
Ŗ "!¢

0’’’’’’’’’
 Æ
*__inference_physicalnn_layer_call_fn_416804576b¢_
X¢U
KH
&#
input_sp’’’’’’’’’``

input_rs’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’Æ
*__inference_physicalnn_layer_call_fn_417464576b¢_
X¢U
KH
&#
input_sp’’’’’’’’’``

input_rs’’’’’’’’’
p

 
Ŗ "’’’’’’’’’Æ
*__inference_physicalnn_layer_call_fn_418244576b¢_
X¢U
KH
&#
inputs/0’’’’’’’’’``

inputs/1’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’Æ
*__inference_physicalnn_layer_call_fn_418384576b¢_
X¢U
KH
&#
inputs/0’’’’’’’’’``

inputs/1’’’’’’’’’
p

 
Ŗ "’’’’’’’’’¤
H__inference_relu_residual_layer_call_and_return_conditional_losses_39061X/¢,
%¢"
 
inputs’’’’’’’’’`
Ŗ "%¢"

0’’’’’’’’’`
 |
-__inference_relu_residual_layer_call_fn_38355K/¢,
%¢"
 
inputs’’’’’’’’’`
Ŗ "’’’’’’’’’`
E__inference_s_function_layer_call_and_return_conditional_losses_38736T76+¢(
!¢

inputs’’’’’’’’’
Ŗ "!¢

0’’’’’’’’’
 u
*__inference_s_function_layer_call_fn_38711G76+¢(
!¢

inputs’’’’’’’’’
Ŗ "’’’’’’’’’
D__inference_shift_sns_layer_call_and_return_conditional_losses_39056W4+¢(
!¢

inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’`
 w
)__inference_shift_sns_layer_call_fn_40217J4+¢(
!¢

inputs’’’’’’’’’
Ŗ "’’’’’’’’’`Ī
#__inference_signature_wrapper_41810¦4576m¢j
¢ 
cŖ`
*
input_rs
input_rs’’’’’’’’’
2
input_sp&#
input_sp’’’’’’’’’``"/Ŗ,
*
any_of_3
any_of_3’’’’’’’’’§
C__inference_spectrum_layer_call_and_return_conditional_losses_38317`7¢4
-¢*
(%
inputs’’’’’’’’’` 
Ŗ "%¢"

0’’’’’’’’’`
 
(__inference_spectrum_layer_call_fn_38645S7¢4
-¢*
(%
inputs’’’’’’’’’` 
Ŗ "’’’’’’’’’`
@__inference_split_layer_call_and_return_conditional_losses_39597¾7¢4
-¢*
(%
inputs’’’’’’’’’``
Ŗ "¢
xu
%"
0/0’’’’’’’’’` 
%"
0/1’’’’’’’’’` 
%"
0/2’’’’’’’’’` 
 ×
%__inference_split_layer_call_fn_39572­7¢4
-¢*
(%
inputs’’’’’’’’’``
Ŗ "ro
# 
0’’’’’’’’’` 
# 
1’’’’’’’’’` 
# 
2’’’’’’’’’` Ė
C__inference_subtract_layer_call_and_return_conditional_losses_42070Z¢W
P¢M
KH
"
inputs/0’’’’’’’’’`
"
inputs/1’’’’’’’’’`
Ŗ "%¢"

0’’’’’’’’’`
 ¢
(__inference_subtract_layer_call_fn_42064vZ¢W
P¢M
KH
"
inputs/0’’’’’’’’’`
"
inputs/1’’’’’’’’’`
Ŗ "’’’’’’’’’`¢
G__inference_weighted_sum_layer_call_and_return_conditional_losses_38725W5/¢,
%¢"
 
inputs’’’’’’’’’`
Ŗ "!¢

0’’’’’’’’’
 z
,__inference_weighted_sum_layer_call_fn_39081J5/¢,
%¢"
 
inputs’’’’’’’’’`
Ŗ "’’’’’’’’’