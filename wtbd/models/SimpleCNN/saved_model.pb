Ý&
Å©
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
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
ú
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%·Ñ8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
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
?
Mul
x"T
y"T
z"T"
Ttype:
2	
0
Neg
x"T
y"T"
Ttype:
2
	
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
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
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
0
Sigmoid
x"T
y"T"
Ttype:

2
-
Sqrt
x"T
y"T"
Ttype:

2
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.5.02unknown8è
d
meanVarHandleOp*
_output_shapes
: *
dtype0*
shape
:``*
shared_namemean
]
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes

:``*
dtype0
l
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape
:``*
shared_name
variance
e
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes

:``*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0	
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
|
fc_batchnorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namefc_batchnorm/gamma
u
&fc_batchnorm/gamma/Read/ReadVariableOpReadVariableOpfc_batchnorm/gamma*
_output_shapes
:*
dtype0
z
fc_batchnorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namefc_batchnorm/beta
s
%fc_batchnorm/beta/Read/ReadVariableOpReadVariableOpfc_batchnorm/beta*
_output_shapes
:*
dtype0

fc_batchnorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namefc_batchnorm/moving_mean

,fc_batchnorm/moving_mean/Read/ReadVariableOpReadVariableOpfc_batchnorm/moving_mean*
_output_shapes
:*
dtype0

fc_batchnorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namefc_batchnorm/moving_variance

0fc_batchnorm/moving_variance/Read/ReadVariableOpReadVariableOpfc_batchnorm/moving_variance*
_output_shapes
:*
dtype0
t
fc_prelu/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namefc_prelu/alpha
m
"fc_prelu/alpha/Read/ReadVariableOpReadVariableOpfc_prelu/alpha*
_output_shapes
:*
dtype0
|
dense_out/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_out/kernel
u
$dense_out/kernel/Read/ReadVariableOpReadVariableOpdense_out/kernel*
_output_shapes

:*
dtype0
t
dense_out/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_out/bias
m
"dense_out/bias/Read/ReadVariableOpReadVariableOpdense_out/bias*
_output_shapes
:*
dtype0

blocks/b1/conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameblocks/b1/conv/kernel

)blocks/b1/conv/kernel/Read/ReadVariableOpReadVariableOpblocks/b1/conv/kernel*&
_output_shapes
:*
dtype0
~
blocks/b1/conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblocks/b1/conv/bias
w
'blocks/b1/conv/bias/Read/ReadVariableOpReadVariableOpblocks/b1/conv/bias*
_output_shapes
:*
dtype0

blocks/b1/batchnorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameblocks/b1/batchnorm/gamma

-blocks/b1/batchnorm/gamma/Read/ReadVariableOpReadVariableOpblocks/b1/batchnorm/gamma*
_output_shapes
:*
dtype0

blocks/b1/batchnorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameblocks/b1/batchnorm/beta

,blocks/b1/batchnorm/beta/Read/ReadVariableOpReadVariableOpblocks/b1/batchnorm/beta*
_output_shapes
:*
dtype0

blocks/b1/batchnorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!blocks/b1/batchnorm/moving_mean

3blocks/b1/batchnorm/moving_mean/Read/ReadVariableOpReadVariableOpblocks/b1/batchnorm/moving_mean*
_output_shapes
:*
dtype0

#blocks/b1/batchnorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#blocks/b1/batchnorm/moving_variance

7blocks/b1/batchnorm/moving_variance/Read/ReadVariableOpReadVariableOp#blocks/b1/batchnorm/moving_variance*
_output_shapes
:*
dtype0

blocks/b1/prelu/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*&
shared_nameblocks/b1/prelu/alpha

)blocks/b1/prelu/alpha/Read/ReadVariableOpReadVariableOpblocks/b1/prelu/alpha*"
_output_shapes
:`*
dtype0

blocks/b2/conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameblocks/b2/conv/kernel

)blocks/b2/conv/kernel/Read/ReadVariableOpReadVariableOpblocks/b2/conv/kernel*&
_output_shapes
:*
dtype0
~
blocks/b2/conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblocks/b2/conv/bias
w
'blocks/b2/conv/bias/Read/ReadVariableOpReadVariableOpblocks/b2/conv/bias*
_output_shapes
:*
dtype0

blocks/b2/batchnorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameblocks/b2/batchnorm/gamma

-blocks/b2/batchnorm/gamma/Read/ReadVariableOpReadVariableOpblocks/b2/batchnorm/gamma*
_output_shapes
:*
dtype0

blocks/b2/batchnorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameblocks/b2/batchnorm/beta

,blocks/b2/batchnorm/beta/Read/ReadVariableOpReadVariableOpblocks/b2/batchnorm/beta*
_output_shapes
:*
dtype0

blocks/b2/batchnorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!blocks/b2/batchnorm/moving_mean

3blocks/b2/batchnorm/moving_mean/Read/ReadVariableOpReadVariableOpblocks/b2/batchnorm/moving_mean*
_output_shapes
:*
dtype0

#blocks/b2/batchnorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#blocks/b2/batchnorm/moving_variance

7blocks/b2/batchnorm/moving_variance/Read/ReadVariableOpReadVariableOp#blocks/b2/batchnorm/moving_variance*
_output_shapes
:*
dtype0

blocks/b2/prelu/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*&
shared_nameblocks/b2/prelu/alpha

)blocks/b2/prelu/alpha/Read/ReadVariableOpReadVariableOpblocks/b2/prelu/alpha*"
_output_shapes
:0*
dtype0

blocks/b3/conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameblocks/b3/conv/kernel

)blocks/b3/conv/kernel/Read/ReadVariableOpReadVariableOpblocks/b3/conv/kernel*&
_output_shapes
:*
dtype0
~
blocks/b3/conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblocks/b3/conv/bias
w
'blocks/b3/conv/bias/Read/ReadVariableOpReadVariableOpblocks/b3/conv/bias*
_output_shapes
:*
dtype0

blocks/b3/batchnorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameblocks/b3/batchnorm/gamma

-blocks/b3/batchnorm/gamma/Read/ReadVariableOpReadVariableOpblocks/b3/batchnorm/gamma*
_output_shapes
:*
dtype0

blocks/b3/batchnorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameblocks/b3/batchnorm/beta

,blocks/b3/batchnorm/beta/Read/ReadVariableOpReadVariableOpblocks/b3/batchnorm/beta*
_output_shapes
:*
dtype0

blocks/b3/batchnorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!blocks/b3/batchnorm/moving_mean

3blocks/b3/batchnorm/moving_mean/Read/ReadVariableOpReadVariableOpblocks/b3/batchnorm/moving_mean*
_output_shapes
:*
dtype0

#blocks/b3/batchnorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#blocks/b3/batchnorm/moving_variance

7blocks/b3/batchnorm/moving_variance/Read/ReadVariableOpReadVariableOp#blocks/b3/batchnorm/moving_variance*
_output_shapes
:*
dtype0

blocks/b3/prelu/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameblocks/b3/prelu/alpha

)blocks/b3/prelu/alpha/Read/ReadVariableOpReadVariableOpblocks/b3/prelu/alpha*"
_output_shapes
:*
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
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
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
{
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*×z
valueÍzBÊz BÃz
Â
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
	optimizer
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api

signatures
%
#_self_saveable_object_factories
»

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean
variance
	count
state_variables
#_self_saveable_object_factories
	keras_api
w
#_self_saveable_object_factories
	variables
 regularization_losses
!trainable_variables
"	keras_api
ì
#layer_with_weights-0
#layer-0
$layer_with_weights-1
$layer-1
%layer_with_weights-2
%layer-2
#&_self_saveable_object_factories
'	variables
(regularization_losses
)trainable_variables
*	keras_api
w
#+_self_saveable_object_factories
,	variables
-regularization_losses
.trainable_variables
/	keras_api


0kernel
1bias
#2_self_saveable_object_factories
3	variables
4regularization_losses
5trainable_variables
6	keras_api
¼
7axis
	8gamma
9beta
:moving_mean
;moving_variance
#<_self_saveable_object_factories
=	variables
>regularization_losses
?trainable_variables
@	keras_api
w
#A_self_saveable_object_factories
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api

Fshared_axes
	Galpha
#H_self_saveable_object_factories
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api


Mkernel
Nbias
#O_self_saveable_object_factories
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
w
#T_self_saveable_object_factories
U	variables
Vregularization_losses
Wtrainable_variables
X	keras_api
 
 
þ
0
1
2
Y3
Z4
[5
\6
]7
^8
_9
`10
a11
b12
c13
d14
e15
f16
g17
h18
i19
j20
k21
l22
m23
024
125
826
927
:28
;29
G30
M31
N32
 
¦
Y0
Z1
[2
\3
_4
`5
a6
b7
c8
f9
g10
h11
i12
j13
m14
015
116
817
918
G19
M20
N21
­
	variables
regularization_losses
nmetrics
trainable_variables
olayer_metrics

players
qnon_trainable_variables
rlayer_regularization_losses
 
 
 
 
 
 
NL
VARIABLE_VALUEmean4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEvariance8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEcount5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUE
#
mean
variance
	count
 
 
 
 
 
 
­
	variables
 regularization_losses
smetrics
!trainable_variables
tlayer_metrics

ulayers
vnon_trainable_variables
wlayer_regularization_losses
ú
xlayer_with_weights-0
xlayer-0
ylayer_with_weights-1
ylayer-1
zlayer_with_weights-2
zlayer-2
{layer-3
#|_self_saveable_object_factories
}	variables
~regularization_losses
trainable_variables
	keras_api

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
$_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
$_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
 

Y0
Z1
[2
\3
]4
^5
_6
`7
a8
b9
c10
d11
e12
f13
g14
h15
i16
j17
k18
l19
m20
 
n
Y0
Z1
[2
\3
_4
`5
a6
b7
c8
f9
g10
h11
i12
j13
m14
²
'	variables
(regularization_losses
metrics
)trainable_variables
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
 
 
 
 
²
,	variables
-regularization_losses
metrics
.trainable_variables
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

00
11
 

00
11
²
3	variables
4regularization_losses
metrics
5trainable_variables
layer_metrics
layers
 non_trainable_variables
 ¡layer_regularization_losses
 
][
VARIABLE_VALUEfc_batchnorm/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEfc_batchnorm/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEfc_batchnorm/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEfc_batchnorm/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

80
91
:2
;3
 

80
91
²
=	variables
>regularization_losses
¢metrics
?trainable_variables
£layer_metrics
¤layers
¥non_trainable_variables
 ¦layer_regularization_losses
 
 
 
 
²
B	variables
Cregularization_losses
§metrics
Dtrainable_variables
¨layer_metrics
©layers
ªnon_trainable_variables
 «layer_regularization_losses
 
YW
VARIABLE_VALUEfc_prelu/alpha5layer_with_weights-4/alpha/.ATTRIBUTES/VARIABLE_VALUE
 

G0
 

G0
²
I	variables
Jregularization_losses
¬metrics
Ktrainable_variables
­layer_metrics
®layers
¯non_trainable_variables
 °layer_regularization_losses
\Z
VARIABLE_VALUEdense_out/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_out/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

M0
N1
 

M0
N1
²
P	variables
Qregularization_losses
±metrics
Rtrainable_variables
²layer_metrics
³layers
´non_trainable_variables
 µlayer_regularization_losses
 
 
 
 
²
U	variables
Vregularization_losses
¶metrics
Wtrainable_variables
·layer_metrics
¸layers
¹non_trainable_variables
 ºlayer_regularization_losses
QO
VARIABLE_VALUEblocks/b1/conv/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblocks/b1/conv/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEblocks/b1/batchnorm/gamma&variables/5/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEblocks/b1/batchnorm/beta&variables/6/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblocks/b1/batchnorm/moving_mean&variables/7/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#blocks/b1/batchnorm/moving_variance&variables/8/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEblocks/b1/prelu/alpha&variables/9/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEblocks/b2/conv/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblocks/b2/conv/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEblocks/b2/batchnorm/gamma'variables/12/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEblocks/b2/batchnorm/beta'variables/13/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblocks/b2/batchnorm/moving_mean'variables/14/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#blocks/b2/batchnorm/moving_variance'variables/15/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEblocks/b2/prelu/alpha'variables/16/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEblocks/b3/conv/kernel'variables/17/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblocks/b3/conv/bias'variables/18/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEblocks/b3/batchnorm/gamma'variables/19/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEblocks/b3/batchnorm/beta'variables/20/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblocks/b3/batchnorm/moving_mean'variables/21/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#blocks/b3/batchnorm/moving_variance'variables/22/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEblocks/b3/prelu/alpha'variables/23/.ATTRIBUTES/VARIABLE_VALUE
 
»0
¼1
½2
¾3
 
N
0
1
2
3
4
5
6
7
	8

9
10
N
0
1
2
]3
^4
d5
e6
k7
l8
:9
;10
 
 
 
 
 
 


Ykernel
Zbias
$¿_self_saveable_object_factories
À	variables
Áregularization_losses
Âtrainable_variables
Ã	keras_api
Â
	Äaxis
	[gamma
\beta
]moving_mean
^moving_variance
$Å_self_saveable_object_factories
Æ	variables
Çregularization_losses
Ètrainable_variables
É	keras_api

Êshared_axes
	_alpha
$Ë_self_saveable_object_factories
Ì	variables
Íregularization_losses
Îtrainable_variables
Ï	keras_api
|
$Ð_self_saveable_object_factories
Ñ	variables
Òregularization_losses
Ótrainable_variables
Ô	keras_api
 
1
Y0
Z1
[2
\3
]4
^5
_6
 
#
Y0
Z1
[2
\3
_4
²
}	variables
~regularization_losses
Õmetrics
trainable_variables
Ölayer_metrics
×layers
Ønon_trainable_variables
 Ùlayer_regularization_losses


`kernel
abias
$Ú_self_saveable_object_factories
Û	variables
Üregularization_losses
Ýtrainable_variables
Þ	keras_api
Â
	ßaxis
	bgamma
cbeta
dmoving_mean
emoving_variance
$à_self_saveable_object_factories
á	variables
âregularization_losses
ãtrainable_variables
ä	keras_api

åshared_axes
	falpha
$æ_self_saveable_object_factories
ç	variables
èregularization_losses
étrainable_variables
ê	keras_api
|
$ë_self_saveable_object_factories
ì	variables
íregularization_losses
îtrainable_variables
ï	keras_api
 
1
`0
a1
b2
c3
d4
e5
f6
 
#
`0
a1
b2
c3
f4
µ
	variables
regularization_losses
ðmetrics
trainable_variables
ñlayer_metrics
òlayers
ónon_trainable_variables
 ôlayer_regularization_losses


gkernel
hbias
$õ_self_saveable_object_factories
ö	variables
÷regularization_losses
øtrainable_variables
ù	keras_api
Â
	úaxis
	igamma
jbeta
kmoving_mean
lmoving_variance
$û_self_saveable_object_factories
ü	variables
ýregularization_losses
þtrainable_variables
ÿ	keras_api

shared_axes
	malpha
$_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
|
$_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
 
1
g0
h1
i2
j3
k4
l5
m6
 
#
g0
h1
i2
j3
m4
µ
	variables
regularization_losses
metrics
trainable_variables
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
 
 

#0
$1
%2
*
]0
^1
d2
e3
k4
l5
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

:0
;1
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
8

total

count
	variables
	keras_api
I

total

count

_fn_kwargs
	variables
	keras_api
\

thresholds
true_positives
false_negatives
	variables
	keras_api
\

thresholds
true_positives
 false_positives
¡	variables
¢	keras_api
 

Y0
Z1
 

Y0
Z1
µ
À	variables
Áregularization_losses
£metrics
Âtrainable_variables
¤layer_metrics
¥layers
¦non_trainable_variables
 §layer_regularization_losses
 
 

[0
\1
]2
^3
 

[0
\1
µ
Æ	variables
Çregularization_losses
¨metrics
Ètrainable_variables
©layer_metrics
ªlayers
«non_trainable_variables
 ¬layer_regularization_losses
 
 

_0
 

_0
µ
Ì	variables
Íregularization_losses
­metrics
Îtrainable_variables
®layer_metrics
¯layers
°non_trainable_variables
 ±layer_regularization_losses
 
 
 
 
µ
Ñ	variables
Òregularization_losses
²metrics
Ótrainable_variables
³layer_metrics
´layers
µnon_trainable_variables
 ¶layer_regularization_losses
 
 

x0
y1
z2
{3

]0
^1
 
 

`0
a1
 

`0
a1
µ
Û	variables
Üregularization_losses
·metrics
Ýtrainable_variables
¸layer_metrics
¹layers
ºnon_trainable_variables
 »layer_regularization_losses
 
 

b0
c1
d2
e3
 

b0
c1
µ
á	variables
âregularization_losses
¼metrics
ãtrainable_variables
½layer_metrics
¾layers
¿non_trainable_variables
 Àlayer_regularization_losses
 
 

f0
 

f0
µ
ç	variables
èregularization_losses
Ámetrics
étrainable_variables
Âlayer_metrics
Ãlayers
Änon_trainable_variables
 Ålayer_regularization_losses
 
 
 
 
µ
ì	variables
íregularization_losses
Æmetrics
îtrainable_variables
Çlayer_metrics
Èlayers
Énon_trainable_variables
 Êlayer_regularization_losses
 
 
 
0
1
2
3

d0
e1
 
 

g0
h1
 

g0
h1
µ
ö	variables
÷regularization_losses
Ëmetrics
øtrainable_variables
Ìlayer_metrics
Ílayers
Înon_trainable_variables
 Ïlayer_regularization_losses
 
 

i0
j1
k2
l3
 

i0
j1
µ
ü	variables
ýregularization_losses
Ðmetrics
þtrainable_variables
Ñlayer_metrics
Òlayers
Ónon_trainable_variables
 Ôlayer_regularization_losses
 
 

m0
 

m0
µ
	variables
regularization_losses
Õmetrics
trainable_variables
Ölayer_metrics
×layers
Ønon_trainable_variables
 Ùlayer_regularization_losses
 
 
 
 
µ
	variables
regularization_losses
Úmetrics
trainable_variables
Ûlayer_metrics
Ülayers
Ýnon_trainable_variables
 Þlayer_regularization_losses
 
 
 
0
1
2
3

k0
l1
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables
 
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUE

0
 1

¡	variables
 
 
 
 
 
 
 
 

]0
^1
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

d0
e1
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

k0
l1
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

serving_default_input_spPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ``
ê
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_spmeanvarianceblocks/b1/conv/kernelblocks/b1/conv/biasblocks/b1/batchnorm/gammablocks/b1/batchnorm/betablocks/b1/batchnorm/moving_mean#blocks/b1/batchnorm/moving_varianceblocks/b1/prelu/alphablocks/b2/conv/kernelblocks/b2/conv/biasblocks/b2/batchnorm/gammablocks/b2/batchnorm/betablocks/b2/batchnorm/moving_mean#blocks/b2/batchnorm/moving_varianceblocks/b2/prelu/alphablocks/b3/conv/kernelblocks/b3/conv/biasblocks/b3/batchnorm/gammablocks/b3/batchnorm/betablocks/b3/batchnorm/moving_mean#blocks/b3/batchnorm/moving_varianceblocks/b3/prelu/alphadense/kernel
dense/biasfc_batchnorm/moving_variancefc_batchnorm/gammafc_batchnorm/moving_meanfc_batchnorm/betafc_prelu/alphadense_out/kerneldense_out/bias*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_51107
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ü
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp&fc_batchnorm/gamma/Read/ReadVariableOp%fc_batchnorm/beta/Read/ReadVariableOp,fc_batchnorm/moving_mean/Read/ReadVariableOp0fc_batchnorm/moving_variance/Read/ReadVariableOp"fc_prelu/alpha/Read/ReadVariableOp$dense_out/kernel/Read/ReadVariableOp"dense_out/bias/Read/ReadVariableOp)blocks/b1/conv/kernel/Read/ReadVariableOp'blocks/b1/conv/bias/Read/ReadVariableOp-blocks/b1/batchnorm/gamma/Read/ReadVariableOp,blocks/b1/batchnorm/beta/Read/ReadVariableOp3blocks/b1/batchnorm/moving_mean/Read/ReadVariableOp7blocks/b1/batchnorm/moving_variance/Read/ReadVariableOp)blocks/b1/prelu/alpha/Read/ReadVariableOp)blocks/b2/conv/kernel/Read/ReadVariableOp'blocks/b2/conv/bias/Read/ReadVariableOp-blocks/b2/batchnorm/gamma/Read/ReadVariableOp,blocks/b2/batchnorm/beta/Read/ReadVariableOp3blocks/b2/batchnorm/moving_mean/Read/ReadVariableOp7blocks/b2/batchnorm/moving_variance/Read/ReadVariableOp)blocks/b2/prelu/alpha/Read/ReadVariableOp)blocks/b3/conv/kernel/Read/ReadVariableOp'blocks/b3/conv/bias/Read/ReadVariableOp-blocks/b3/batchnorm/gamma/Read/ReadVariableOp,blocks/b3/batchnorm/beta/Read/ReadVariableOp3blocks/b3/batchnorm/moving_mean/Read/ReadVariableOp7blocks/b3/batchnorm/moving_variance/Read/ReadVariableOp)blocks/b3/prelu/alpha/Read/ReadVariableOptotal/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_2/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp#false_positives/Read/ReadVariableOpConst*6
Tin/
-2+	*
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
__inference__traced_save_52848
£	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecountdense/kernel
dense/biasfc_batchnorm/gammafc_batchnorm/betafc_batchnorm/moving_meanfc_batchnorm/moving_variancefc_prelu/alphadense_out/kerneldense_out/biasblocks/b1/conv/kernelblocks/b1/conv/biasblocks/b1/batchnorm/gammablocks/b1/batchnorm/betablocks/b1/batchnorm/moving_mean#blocks/b1/batchnorm/moving_varianceblocks/b1/prelu/alphablocks/b2/conv/kernelblocks/b2/conv/biasblocks/b2/batchnorm/gammablocks/b2/batchnorm/betablocks/b2/batchnorm/moving_mean#blocks/b2/batchnorm/moving_varianceblocks/b2/prelu/alphablocks/b3/conv/kernelblocks/b3/conv/biasblocks/b3/batchnorm/gammablocks/b3/batchnorm/betablocks/b3/batchnorm/moving_mean#blocks/b3/batchnorm/moving_varianceblocks/b3/prelu/alphatotalcount_1total_1count_2true_positivesfalse_negativestrue_positives_1false_positives*5
Tin.
,2**
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
!__inference__traced_restore_52981ø


D__inference_blocks/b1_layer_call_and_return_conditional_losses_48953
blocks_b1_conv_input.
blocks_b1_conv_48934:"
blocks_b1_conv_48936:'
blocks_b1_batchnorm_48939:'
blocks_b1_batchnorm_48941:'
blocks_b1_batchnorm_48943:'
blocks_b1_batchnorm_48945:+
blocks_b1_prelu_48948:`
identity¢+blocks/b1/batchnorm/StatefulPartitionedCall¢&blocks/b1/conv/StatefulPartitionedCall¢'blocks/b1/prelu/StatefulPartitionedCallÈ
&blocks/b1/conv/StatefulPartitionedCallStatefulPartitionedCallblocks_b1_conv_inputblocks_b1_conv_48934blocks_b1_conv_48936*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_blocks/b1/conv_layer_call_and_return_conditional_losses_487212(
&blocks/b1/conv/StatefulPartitionedCall´
+blocks/b1/batchnorm/StatefulPartitionedCallStatefulPartitionedCall/blocks/b1/conv/StatefulPartitionedCall:output:0blocks_b1_batchnorm_48939blocks_b1_batchnorm_48941blocks_b1_batchnorm_48943blocks_b1_batchnorm_48945*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_blocks/b1/batchnorm_layer_call_and_return_conditional_losses_488092-
+blocks/b1/batchnorm/StatefulPartitionedCallÔ
'blocks/b1/prelu/StatefulPartitionedCallStatefulPartitionedCall4blocks/b1/batchnorm/StatefulPartitionedCall:output:0blocks_b1_prelu_48948*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_blocks/b1/prelu_layer_call_and_return_conditional_losses_486842)
'blocks/b1/prelu/StatefulPartitionedCall£
!blocks/b1/maxpool/PartitionedCallPartitionedCall0blocks/b1/prelu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_blocks/b1/maxpool_layer_call_and_return_conditional_losses_486982#
!blocks/b1/maxpool/PartitionedCall
IdentityIdentity*blocks/b1/maxpool/PartitionedCall:output:0,^blocks/b1/batchnorm/StatefulPartitionedCall'^blocks/b1/conv/StatefulPartitionedCall(^blocks/b1/prelu/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ``: : : : : : : 2Z
+blocks/b1/batchnorm/StatefulPartitionedCall+blocks/b1/batchnorm/StatefulPartitionedCall2P
&blocks/b1/conv/StatefulPartitionedCall&blocks/b1/conv/StatefulPartitionedCall2R
'blocks/b1/prelu/StatefulPartitionedCall'blocks/b1/prelu/StatefulPartitionedCall:e a
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
.
_user_specified_nameblocks/b1/conv_input
 +
å
D__inference_blocks/b2_layer_call_and_return_conditional_losses_52136

inputsG
-blocks_b2_conv_conv2d_readvariableop_resource:<
.blocks_b2_conv_biasadd_readvariableop_resource:9
+blocks_b2_batchnorm_readvariableop_resource:;
-blocks_b2_batchnorm_readvariableop_1_resource:J
<blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_resource:L
>blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_1_resource:=
'blocks_b2_prelu_readvariableop_resource:0
identity¢3blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp¢5blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1¢"blocks/b2/batchnorm/ReadVariableOp¢$blocks/b2/batchnorm/ReadVariableOp_1¢%blocks/b2/conv/BiasAdd/ReadVariableOp¢$blocks/b2/conv/Conv2D/ReadVariableOp¢blocks/b2/prelu/ReadVariableOpÂ
$blocks/b2/conv/Conv2D/ReadVariableOpReadVariableOp-blocks_b2_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$blocks/b2/conv/Conv2D/ReadVariableOpÐ
blocks/b2/conv/Conv2DConv2Dinputs,blocks/b2/conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*
paddingSAME*
strides
2
blocks/b2/conv/Conv2D¹
%blocks/b2/conv/BiasAdd/ReadVariableOpReadVariableOp.blocks_b2_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%blocks/b2/conv/BiasAdd/ReadVariableOpÄ
blocks/b2/conv/BiasAddBiasAddblocks/b2/conv/Conv2D:output:0-blocks/b2/conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002
blocks/b2/conv/BiasAdd°
"blocks/b2/batchnorm/ReadVariableOpReadVariableOp+blocks_b2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02$
"blocks/b2/batchnorm/ReadVariableOp¶
$blocks/b2/batchnorm/ReadVariableOp_1ReadVariableOp-blocks_b2_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$blocks/b2/batchnorm/ReadVariableOp_1ã
3blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOpReadVariableOp<blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOpé
5blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1Û
$blocks/b2/batchnorm/FusedBatchNormV3FusedBatchNormV3blocks/b2/conv/BiasAdd:output:0*blocks/b2/batchnorm/ReadVariableOp:value:0,blocks/b2/batchnorm/ReadVariableOp_1:value:0;blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp:value:0=blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00:::::*
epsilon%o:*
is_training( 2&
$blocks/b2/batchnorm/FusedBatchNormV3
blocks/b2/prelu/ReluRelu(blocks/b2/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002
blocks/b2/prelu/Relu¬
blocks/b2/prelu/ReadVariableOpReadVariableOp'blocks_b2_prelu_readvariableop_resource*"
_output_shapes
:0*
dtype02 
blocks/b2/prelu/ReadVariableOp
blocks/b2/prelu/NegNeg&blocks/b2/prelu/ReadVariableOp:value:0*
T0*"
_output_shapes
:02
blocks/b2/prelu/Neg
blocks/b2/prelu/Neg_1Neg(blocks/b2/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002
blocks/b2/prelu/Neg_1
blocks/b2/prelu/Relu_1Relublocks/b2/prelu/Neg_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002
blocks/b2/prelu/Relu_1ª
blocks/b2/prelu/mulMulblocks/b2/prelu/Neg:y:0$blocks/b2/prelu/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002
blocks/b2/prelu/mulª
blocks/b2/prelu/addAddV2"blocks/b2/prelu/Relu:activations:0blocks/b2/prelu/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002
blocks/b2/prelu/addÆ
blocks/b2/maxpool/MaxPoolMaxPoolblocks/b2/prelu/add:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2
blocks/b2/maxpool/MaxPool¨
IdentityIdentity"blocks/b2/maxpool/MaxPool:output:04^blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp6^blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1#^blocks/b2/batchnorm/ReadVariableOp%^blocks/b2/batchnorm/ReadVariableOp_1&^blocks/b2/conv/BiasAdd/ReadVariableOp%^blocks/b2/conv/Conv2D/ReadVariableOp^blocks/b2/prelu/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ00: : : : : : : 2j
3blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp3blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp2n
5blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_15blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_12H
"blocks/b2/batchnorm/ReadVariableOp"blocks/b2/batchnorm/ReadVariableOp2L
$blocks/b2/batchnorm/ReadVariableOp_1$blocks/b2/batchnorm/ReadVariableOp_12N
%blocks/b2/conv/BiasAdd/ReadVariableOp%blocks/b2/conv/BiasAdd/ReadVariableOp2L
$blocks/b2/conv/Conv2D/ReadVariableOp$blocks/b2/conv/Conv2D/ReadVariableOp2@
blocks/b2/prelu/ReadVariableOpblocks/b2/prelu/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
 
_user_specified_nameinputs
À	
±
)__inference_blocks/b3_layer_call_fn_49592
blocks_b3_conv_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
identity¢StatefulPartitionedCallÎ
StatefulPartitionedCallStatefulPartitionedCallblocks_b3_conv_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_blocks/b3_layer_call_and_return_conditional_losses_495752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.
_user_specified_nameblocks/b3/conv_input
¤
Ý
#__inference_signature_wrapper_51107
input_sp
unknown:``
	unknown_0:``#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:`#
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13: 

unknown_14:0$

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20: 

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:
identity¢StatefulPartitionedCallð
StatefulPartitionedCallStatefulPartitionedCallinput_spunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_485452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
"
_user_specified_name
input_sp
°

A__inference_blocks_layer_call_and_return_conditional_losses_49822

inputs)
blocks_b1_49776:
blocks_b1_49778:
blocks_b1_49780:
blocks_b1_49782:
blocks_b1_49784:
blocks_b1_49786:%
blocks_b1_49788:`)
blocks_b2_49791:
blocks_b2_49793:
blocks_b2_49795:
blocks_b2_49797:
blocks_b2_49799:
blocks_b2_49801:%
blocks_b2_49803:0)
blocks_b3_49806:
blocks_b3_49808:
blocks_b3_49810:
blocks_b3_49812:
blocks_b3_49814:
blocks_b3_49816:%
blocks_b3_49818:
identity¢!blocks/b1/StatefulPartitionedCall¢!blocks/b2/StatefulPartitionedCall¢!blocks/b3/StatefulPartitionedCall
!blocks/b1/StatefulPartitionedCallStatefulPartitionedCallinputsblocks_b1_49776blocks_b1_49778blocks_b1_49780blocks_b1_49782blocks_b1_49784blocks_b1_49786blocks_b1_49788*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_blocks/b1_layer_call_and_return_conditional_losses_487592#
!blocks/b1/StatefulPartitionedCall¤
!blocks/b2/StatefulPartitionedCallStatefulPartitionedCall*blocks/b1/StatefulPartitionedCall:output:0blocks_b2_49791blocks_b2_49793blocks_b2_49795blocks_b2_49797blocks_b2_49799blocks_b2_49801blocks_b2_49803*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_blocks/b2_layer_call_and_return_conditional_losses_491672#
!blocks/b2/StatefulPartitionedCall¤
!blocks/b3/StatefulPartitionedCallStatefulPartitionedCall*blocks/b2/StatefulPartitionedCall:output:0blocks_b3_49806blocks_b3_49808blocks_b3_49810blocks_b3_49812blocks_b3_49814blocks_b3_49816blocks_b3_49818*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_blocks/b3_layer_call_and_return_conditional_losses_495752#
!blocks/b3/StatefulPartitionedCallò
IdentityIdentity*blocks/b3/StatefulPartitionedCall:output:0"^blocks/b1/StatefulPartitionedCall"^blocks/b2/StatefulPartitionedCall"^blocks/b3/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : 2F
!blocks/b1/StatefulPartitionedCall!blocks/b1/StatefulPartitionedCall2F
!blocks/b2/StatefulPartitionedCall!blocks/b2/StatefulPartitionedCall2F
!blocks/b3/StatefulPartitionedCall!blocks/b3/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
Á
F
*__inference_fc_dropout_layer_call_fn_51928

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_fc_dropout_layer_call_and_return_conditional_losses_504462
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À

N__inference_blocks/b3/batchnorm_layer_call_and_return_conditional_losses_49560

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ú
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
£
)__inference_blocks/b1_layer_call_fn_51999

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:`
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_blocks/b1_layer_call_and_return_conditional_losses_488732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ``: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs


D__inference_blocks/b2_layer_call_and_return_conditional_losses_49339
blocks_b2_conv_input.
blocks_b2_conv_49320:"
blocks_b2_conv_49322:'
blocks_b2_batchnorm_49325:'
blocks_b2_batchnorm_49327:'
blocks_b2_batchnorm_49329:'
blocks_b2_batchnorm_49331:+
blocks_b2_prelu_49334:0
identity¢+blocks/b2/batchnorm/StatefulPartitionedCall¢&blocks/b2/conv/StatefulPartitionedCall¢'blocks/b2/prelu/StatefulPartitionedCallÈ
&blocks/b2/conv/StatefulPartitionedCallStatefulPartitionedCallblocks_b2_conv_inputblocks_b2_conv_49320blocks_b2_conv_49322*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_blocks/b2/conv_layer_call_and_return_conditional_losses_491292(
&blocks/b2/conv/StatefulPartitionedCall¶
+blocks/b2/batchnorm/StatefulPartitionedCallStatefulPartitionedCall/blocks/b2/conv/StatefulPartitionedCall:output:0blocks_b2_batchnorm_49325blocks_b2_batchnorm_49327blocks_b2_batchnorm_49329blocks_b2_batchnorm_49331*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_blocks/b2/batchnorm_layer_call_and_return_conditional_losses_491522-
+blocks/b2/batchnorm/StatefulPartitionedCallÔ
'blocks/b2/prelu/StatefulPartitionedCallStatefulPartitionedCall4blocks/b2/batchnorm/StatefulPartitionedCall:output:0blocks_b2_prelu_49334*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_blocks/b2/prelu_layer_call_and_return_conditional_losses_490922)
'blocks/b2/prelu/StatefulPartitionedCall£
!blocks/b2/maxpool/PartitionedCallPartitionedCall0blocks/b2/prelu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_blocks/b2/maxpool_layer_call_and_return_conditional_losses_491062#
!blocks/b2/maxpool/PartitionedCall
IdentityIdentity*blocks/b2/maxpool/PartitionedCall:output:0,^blocks/b2/batchnorm/StatefulPartitionedCall'^blocks/b2/conv/StatefulPartitionedCall(^blocks/b2/prelu/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ00: : : : : : : 2Z
+blocks/b2/batchnorm/StatefulPartitionedCall+blocks/b2/batchnorm/StatefulPartitionedCall2P
&blocks/b2/conv/StatefulPartitionedCall&blocks/b2/conv/StatefulPartitionedCall2R
'blocks/b2/prelu/StatefulPartitionedCall'blocks/b2/prelu/StatefulPartitionedCall:e a
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
.
_user_specified_nameblocks/b2/conv_input
Á
F
*__inference_fc_dropout_layer_call_fn_51933

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_fc_dropout_layer_call_and_return_conditional_losses_505582
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
½
N__inference_blocks/b3/batchnorm_layer_call_and_return_conditional_losses_49427

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 +
å
D__inference_blocks/b3_layer_call_and_return_conditional_losses_52240

inputsG
-blocks_b3_conv_conv2d_readvariableop_resource:<
.blocks_b3_conv_biasadd_readvariableop_resource:9
+blocks_b3_batchnorm_readvariableop_resource:;
-blocks_b3_batchnorm_readvariableop_1_resource:J
<blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_resource:L
>blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_1_resource:=
'blocks_b3_prelu_readvariableop_resource:
identity¢3blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp¢5blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1¢"blocks/b3/batchnorm/ReadVariableOp¢$blocks/b3/batchnorm/ReadVariableOp_1¢%blocks/b3/conv/BiasAdd/ReadVariableOp¢$blocks/b3/conv/Conv2D/ReadVariableOp¢blocks/b3/prelu/ReadVariableOpÂ
$blocks/b3/conv/Conv2D/ReadVariableOpReadVariableOp-blocks_b3_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$blocks/b3/conv/Conv2D/ReadVariableOpÐ
blocks/b3/conv/Conv2DConv2Dinputs,blocks/b3/conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
blocks/b3/conv/Conv2D¹
%blocks/b3/conv/BiasAdd/ReadVariableOpReadVariableOp.blocks_b3_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%blocks/b3/conv/BiasAdd/ReadVariableOpÄ
blocks/b3/conv/BiasAddBiasAddblocks/b3/conv/Conv2D:output:0-blocks/b3/conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
blocks/b3/conv/BiasAdd°
"blocks/b3/batchnorm/ReadVariableOpReadVariableOp+blocks_b3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02$
"blocks/b3/batchnorm/ReadVariableOp¶
$blocks/b3/batchnorm/ReadVariableOp_1ReadVariableOp-blocks_b3_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$blocks/b3/batchnorm/ReadVariableOp_1ã
3blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOpReadVariableOp<blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOpé
5blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1Û
$blocks/b3/batchnorm/FusedBatchNormV3FusedBatchNormV3blocks/b3/conv/BiasAdd:output:0*blocks/b3/batchnorm/ReadVariableOp:value:0,blocks/b3/batchnorm/ReadVariableOp_1:value:0;blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp:value:0=blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2&
$blocks/b3/batchnorm/FusedBatchNormV3
blocks/b3/prelu/ReluRelu(blocks/b3/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
blocks/b3/prelu/Relu¬
blocks/b3/prelu/ReadVariableOpReadVariableOp'blocks_b3_prelu_readvariableop_resource*"
_output_shapes
:*
dtype02 
blocks/b3/prelu/ReadVariableOp
blocks/b3/prelu/NegNeg&blocks/b3/prelu/ReadVariableOp:value:0*
T0*"
_output_shapes
:2
blocks/b3/prelu/Neg
blocks/b3/prelu/Neg_1Neg(blocks/b3/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
blocks/b3/prelu/Neg_1
blocks/b3/prelu/Relu_1Relublocks/b3/prelu/Neg_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
blocks/b3/prelu/Relu_1ª
blocks/b3/prelu/mulMulblocks/b3/prelu/Neg:y:0$blocks/b3/prelu/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
blocks/b3/prelu/mulª
blocks/b3/prelu/addAddV2"blocks/b3/prelu/Relu:activations:0blocks/b3/prelu/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
blocks/b3/prelu/addÆ
blocks/b3/maxpool/MaxPoolMaxPoolblocks/b3/prelu/add:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2
blocks/b3/maxpool/MaxPool¨
IdentityIdentity"blocks/b3/maxpool/MaxPool:output:04^blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp6^blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1#^blocks/b3/batchnorm/ReadVariableOp%^blocks/b3/batchnorm/ReadVariableOp_1&^blocks/b3/conv/BiasAdd/ReadVariableOp%^blocks/b3/conv/Conv2D/ReadVariableOp^blocks/b3/prelu/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : 2j
3blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp3blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp2n
5blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_15blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_12H
"blocks/b3/batchnorm/ReadVariableOp"blocks/b3/batchnorm/ReadVariableOp2L
$blocks/b3/batchnorm/ReadVariableOp_1$blocks/b3/batchnorm/ReadVariableOp_12N
%blocks/b3/conv/BiasAdd/ReadVariableOp%blocks/b3/conv/BiasAdd/ReadVariableOp2L
$blocks/b3/conv/Conv2D/ReadVariableOp$blocks/b3/conv/Conv2D/ReadVariableOp2@
blocks/b3/prelu/ReadVariableOpblocks/b3/prelu/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À
á
)__inference_simplecnn_layer_call_fn_51245

inputs
unknown:``
	unknown_0:``#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:`#
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13: 

unknown_14:0$

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20: 

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*:
_read_only_resource_inputs
	
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_simplecnn_layer_call_and_return_conditional_losses_507282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
¼
½
N__inference_blocks/b1/batchnorm_layer_call_and_return_conditional_losses_48611

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


)__inference_dense_out_layer_call_fn_51951

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_out_layer_call_and_return_conditional_losses_504612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö
°
&__inference_blocks_layer_call_fn_51642

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:`#
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11: 

unknown_12:0$

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18: 

unknown_19:
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19*!
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_blocks_layer_call_and_return_conditional_losses_499652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs

¦
G__inference_fc_batchnorm_layer_call_and_return_conditional_losses_50192

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1Û
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
½
N__inference_blocks/b1/batchnorm_layer_call_and_return_conditional_losses_52380

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
£
)__inference_blocks/b3_layer_call_fn_52188

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_blocks/b3_layer_call_and_return_conditional_losses_495752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô
½
N__inference_blocks/b2/batchnorm_layer_call_and_return_conditional_losses_49217

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1þ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ00: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
 
_user_specified_nameinputs
¼2
±
D__inference_blocks/b1_layer_call_and_return_conditional_losses_52065

inputsG
-blocks_b1_conv_conv2d_readvariableop_resource:<
.blocks_b1_conv_biasadd_readvariableop_resource:9
+blocks_b1_batchnorm_readvariableop_resource:;
-blocks_b1_batchnorm_readvariableop_1_resource:J
<blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_resource:L
>blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_1_resource:=
'blocks_b1_prelu_readvariableop_resource:`
identity¢"blocks/b1/batchnorm/AssignNewValue¢$blocks/b1/batchnorm/AssignNewValue_1¢3blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp¢5blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1¢"blocks/b1/batchnorm/ReadVariableOp¢$blocks/b1/batchnorm/ReadVariableOp_1¢%blocks/b1/conv/BiasAdd/ReadVariableOp¢$blocks/b1/conv/Conv2D/ReadVariableOp¢blocks/b1/prelu/ReadVariableOpÂ
$blocks/b1/conv/Conv2D/ReadVariableOpReadVariableOp-blocks_b1_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$blocks/b1/conv/Conv2D/ReadVariableOpÐ
blocks/b1/conv/Conv2DConv2Dinputs,blocks/b1/conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*
paddingSAME*
strides
2
blocks/b1/conv/Conv2D¹
%blocks/b1/conv/BiasAdd/ReadVariableOpReadVariableOp.blocks_b1_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%blocks/b1/conv/BiasAdd/ReadVariableOpÄ
blocks/b1/conv/BiasAddBiasAddblocks/b1/conv/Conv2D:output:0-blocks/b1/conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
blocks/b1/conv/BiasAdd°
"blocks/b1/batchnorm/ReadVariableOpReadVariableOp+blocks_b1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02$
"blocks/b1/batchnorm/ReadVariableOp¶
$blocks/b1/batchnorm/ReadVariableOp_1ReadVariableOp-blocks_b1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$blocks/b1/batchnorm/ReadVariableOp_1ã
3blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOpReadVariableOp<blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOpé
5blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1é
$blocks/b1/batchnorm/FusedBatchNormV3FusedBatchNormV3blocks/b1/conv/BiasAdd:output:0*blocks/b1/batchnorm/ReadVariableOp:value:0,blocks/b1/batchnorm/ReadVariableOp_1:value:0;blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp:value:0=blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ``:::::*
epsilon%o:*
exponential_avg_factor%
×#<2&
$blocks/b1/batchnorm/FusedBatchNormV3¦
"blocks/b1/batchnorm/AssignNewValueAssignVariableOp<blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_resource1blocks/b1/batchnorm/FusedBatchNormV3:batch_mean:04^blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"blocks/b1/batchnorm/AssignNewValue²
$blocks/b1/batchnorm/AssignNewValue_1AssignVariableOp>blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_1_resource5blocks/b1/batchnorm/FusedBatchNormV3:batch_variance:06^blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$blocks/b1/batchnorm/AssignNewValue_1
blocks/b1/prelu/ReluRelu(blocks/b1/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
blocks/b1/prelu/Relu¬
blocks/b1/prelu/ReadVariableOpReadVariableOp'blocks_b1_prelu_readvariableop_resource*"
_output_shapes
:`*
dtype02 
blocks/b1/prelu/ReadVariableOp
blocks/b1/prelu/NegNeg&blocks/b1/prelu/ReadVariableOp:value:0*
T0*"
_output_shapes
:`2
blocks/b1/prelu/Neg
blocks/b1/prelu/Neg_1Neg(blocks/b1/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
blocks/b1/prelu/Neg_1
blocks/b1/prelu/Relu_1Relublocks/b1/prelu/Neg_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
blocks/b1/prelu/Relu_1ª
blocks/b1/prelu/mulMulblocks/b1/prelu/Neg:y:0$blocks/b1/prelu/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
blocks/b1/prelu/mulª
blocks/b1/prelu/addAddV2"blocks/b1/prelu/Relu:activations:0blocks/b1/prelu/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
blocks/b1/prelu/addÆ
blocks/b1/maxpool/MaxPoolMaxPoolblocks/b1/prelu/add:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*
ksize
*
paddingSAME*
strides
2
blocks/b1/maxpool/MaxPoolô
IdentityIdentity"blocks/b1/maxpool/MaxPool:output:0#^blocks/b1/batchnorm/AssignNewValue%^blocks/b1/batchnorm/AssignNewValue_14^blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp6^blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1#^blocks/b1/batchnorm/ReadVariableOp%^blocks/b1/batchnorm/ReadVariableOp_1&^blocks/b1/conv/BiasAdd/ReadVariableOp%^blocks/b1/conv/Conv2D/ReadVariableOp^blocks/b1/prelu/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ``: : : : : : : 2H
"blocks/b1/batchnorm/AssignNewValue"blocks/b1/batchnorm/AssignNewValue2L
$blocks/b1/batchnorm/AssignNewValue_1$blocks/b1/batchnorm/AssignNewValue_12j
3blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp3blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp2n
5blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_15blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_12H
"blocks/b1/batchnorm/ReadVariableOp"blocks/b1/batchnorm/ReadVariableOp2L
$blocks/b1/batchnorm/ReadVariableOp_1$blocks/b1/batchnorm/ReadVariableOp_12N
%blocks/b1/conv/BiasAdd/ReadVariableOp%blocks/b1/conv/BiasAdd/ReadVariableOp2L
$blocks/b1/conv/Conv2D/ReadVariableOp$blocks/b1/conv/Conv2D/ReadVariableOp2@
blocks/b1/prelu/ReadVariableOpblocks/b1/prelu/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
²B

D__inference_simplecnn_layer_call_and_return_conditional_losses_50728

inputsA
/featurewise_std_reshape_readvariableop_resource:``C
1featurewise_std_reshape_1_readvariableop_resource:``&
blocks_50659:
blocks_50661:
blocks_50663:
blocks_50665:
blocks_50667:
blocks_50669:"
blocks_50671:`&
blocks_50673:
blocks_50675:
blocks_50677:
blocks_50679:
blocks_50681:
blocks_50683:"
blocks_50685:0&
blocks_50687:
blocks_50689:
blocks_50691:
blocks_50693:
blocks_50695:
blocks_50697:"
blocks_50699:
dense_50703:
dense_50705: 
fc_batchnorm_50708: 
fc_batchnorm_50710: 
fc_batchnorm_50712: 
fc_batchnorm_50714:
fc_prelu_50718:!
dense_out_50721:
dense_out_50723:
identity¢blocks/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢!dense_out/StatefulPartitionedCall¢$fc_batchnorm/StatefulPartitionedCall¢ fc_prelu/StatefulPartitionedCall¢&featurewise_std/Reshape/ReadVariableOp¢(featurewise_std/Reshape_1/ReadVariableOpÀ
&featurewise_std/Reshape/ReadVariableOpReadVariableOp/featurewise_std_reshape_readvariableop_resource*
_output_shapes

:``*
dtype02(
&featurewise_std/Reshape/ReadVariableOp
featurewise_std/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   `   `   2
featurewise_std/Reshape/shapeÂ
featurewise_std/ReshapeReshape.featurewise_std/Reshape/ReadVariableOp:value:0&featurewise_std/Reshape/shape:output:0*
T0*"
_output_shapes
:``2
featurewise_std/ReshapeÆ
(featurewise_std/Reshape_1/ReadVariableOpReadVariableOp1featurewise_std_reshape_1_readvariableop_resource*
_output_shapes

:``*
dtype02*
(featurewise_std/Reshape_1/ReadVariableOp
featurewise_std/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   `   `   2!
featurewise_std/Reshape_1/shapeÊ
featurewise_std/Reshape_1Reshape0featurewise_std/Reshape_1/ReadVariableOp:value:0(featurewise_std/Reshape_1/shape:output:0*
T0*"
_output_shapes
:``2
featurewise_std/Reshape_1
featurewise_std/subSubinputs featurewise_std/Reshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
featurewise_std/sub
featurewise_std/SqrtSqrt"featurewise_std/Reshape_1:output:0*
T0*"
_output_shapes
:``2
featurewise_std/Sqrt{
featurewise_std/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32
featurewise_std/Maximum/y¨
featurewise_std/MaximumMaximumfeaturewise_std/Sqrt:y:0"featurewise_std/Maximum/y:output:0*
T0*"
_output_shapes
:``2
featurewise_std/Maximum©
featurewise_std/truedivRealDivfeaturewise_std/sub:z:0featurewise_std/Maximum:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
featurewise_std/truedivî
#insert_channel_axis/PartitionedCallPartitionedCallfeaturewise_std/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_484102%
#insert_channel_axis/PartitionedCallâ
blocks/StatefulPartitionedCallStatefulPartitionedCall,insert_channel_axis/PartitionedCall:output:0blocks_50659blocks_50661blocks_50663blocks_50665blocks_50667blocks_50669blocks_50671blocks_50673blocks_50675blocks_50677blocks_50679blocks_50681blocks_50683blocks_50685blocks_50687blocks_50689blocks_50691blocks_50693blocks_50695blocks_50697blocks_50699*!
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_blocks_layer_call_and_return_conditional_losses_499652 
blocks/StatefulPartitionedCall
globalavgpool/PartitionedCallPartitionedCall'blocks/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_globalavgpool_layer_call_and_return_conditional_losses_501622
globalavgpool/PartitionedCall¥
dense/StatefulPartitionedCallStatefulPartitionedCall&globalavgpool/PartitionedCall:output:0dense_50703dense_50705*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_504262
dense/StatefulPartitionedCallò
$fc_batchnorm/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0fc_batchnorm_50708fc_batchnorm_50710fc_batchnorm_50712fc_batchnorm_50714*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_fc_batchnorm_layer_call_and_return_conditional_losses_502522&
$fc_batchnorm/StatefulPartitionedCall
fc_dropout/PartitionedCallPartitionedCall-fc_batchnorm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_fc_dropout_layer_call_and_return_conditional_losses_505582
fc_dropout/PartitionedCall
 fc_prelu/StatefulPartitionedCallStatefulPartitionedCall#fc_dropout/PartitionedCall:output:0fc_prelu_50718*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_fc_prelu_layer_call_and_return_conditional_losses_503432"
 fc_prelu/StatefulPartitionedCall¼
!dense_out/StatefulPartitionedCallStatefulPartitionedCall)fc_prelu/StatefulPartitionedCall:output:0dense_out_50721dense_out_50723*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_out_layer_call_and_return_conditional_losses_504612#
!dense_out/StatefulPartitionedCallå
probability/PartitionedCallPartitionedCall*dense_out/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_485422
probability/PartitionedCallû
IdentityIdentity$probability/PartitionedCall:output:0^blocks/StatefulPartitionedCall^dense/StatefulPartitionedCall"^dense_out/StatefulPartitionedCall%^fc_batchnorm/StatefulPartitionedCall!^fc_prelu/StatefulPartitionedCall'^featurewise_std/Reshape/ReadVariableOp)^featurewise_std/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
blocks/StatefulPartitionedCallblocks/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!dense_out/StatefulPartitionedCall!dense_out/StatefulPartitionedCall2L
$fc_batchnorm/StatefulPartitionedCall$fc_batchnorm/StatefulPartitionedCall2D
 fc_prelu/StatefulPartitionedCall fc_prelu/StatefulPartitionedCall2P
&featurewise_std/Reshape/ReadVariableOp&featurewise_std/Reshape/ReadVariableOp2T
(featurewise_std/Reshape_1/ReadVariableOp(featurewise_std/Reshape_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs

j
N__inference_insert_channel_axis_layer_call_and_return_conditional_losses_42946

inputs
identityk
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2

ExpandDimso
IdentityIdentityExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ``:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
¾	
±
)__inference_blocks/b3_layer_call_fn_49725
blocks_b3_conv_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
identity¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallblocks_b3_conv_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_blocks/b3_layer_call_and_return_conditional_losses_496892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.
_user_specified_nameblocks/b3/conv_input
à
b
F__inference_probability_layer_call_and_return_conditional_losses_43277

inputs
identityW
SigmoidSigmoidinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
£
.__inference_blocks/b3/conv_layer_call_fn_52568

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_blocks/b3/conv_layer_call_and_return_conditional_losses_495372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë

D__inference_blocks/b2_layer_call_and_return_conditional_losses_49167

inputs.
blocks_b2_conv_49130:"
blocks_b2_conv_49132:'
blocks_b2_batchnorm_49153:'
blocks_b2_batchnorm_49155:'
blocks_b2_batchnorm_49157:'
blocks_b2_batchnorm_49159:+
blocks_b2_prelu_49162:0
identity¢+blocks/b2/batchnorm/StatefulPartitionedCall¢&blocks/b2/conv/StatefulPartitionedCall¢'blocks/b2/prelu/StatefulPartitionedCallº
&blocks/b2/conv/StatefulPartitionedCallStatefulPartitionedCallinputsblocks_b2_conv_49130blocks_b2_conv_49132*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_blocks/b2/conv_layer_call_and_return_conditional_losses_491292(
&blocks/b2/conv/StatefulPartitionedCall¶
+blocks/b2/batchnorm/StatefulPartitionedCallStatefulPartitionedCall/blocks/b2/conv/StatefulPartitionedCall:output:0blocks_b2_batchnorm_49153blocks_b2_batchnorm_49155blocks_b2_batchnorm_49157blocks_b2_batchnorm_49159*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_blocks/b2/batchnorm_layer_call_and_return_conditional_losses_491522-
+blocks/b2/batchnorm/StatefulPartitionedCallÔ
'blocks/b2/prelu/StatefulPartitionedCallStatefulPartitionedCall4blocks/b2/batchnorm/StatefulPartitionedCall:output:0blocks_b2_prelu_49162*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_blocks/b2/prelu_layer_call_and_return_conditional_losses_490922)
'blocks/b2/prelu/StatefulPartitionedCall£
!blocks/b2/maxpool/PartitionedCallPartitionedCall0blocks/b2/prelu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_blocks/b2/maxpool_layer_call_and_return_conditional_losses_491062#
!blocks/b2/maxpool/PartitionedCall
IdentityIdentity*blocks/b2/maxpool/PartitionedCall:output:0,^blocks/b2/batchnorm/StatefulPartitionedCall'^blocks/b2/conv/StatefulPartitionedCall(^blocks/b2/prelu/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ00: : : : : : : 2Z
+blocks/b2/batchnorm/StatefulPartitionedCall+blocks/b2/batchnorm/StatefulPartitionedCall2P
&blocks/b2/conv/StatefulPartitionedCall&blocks/b2/conv/StatefulPartitionedCall2R
'blocks/b2/prelu/StatefulPartitionedCall'blocks/b2/prelu/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
 
_user_specified_nameinputs
µ


I__inference_blocks/b1/conv_layer_call_and_return_conditional_losses_48721

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ``: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
ë

D__inference_blocks/b3_layer_call_and_return_conditional_losses_49575

inputs.
blocks_b3_conv_49538:"
blocks_b3_conv_49540:'
blocks_b3_batchnorm_49561:'
blocks_b3_batchnorm_49563:'
blocks_b3_batchnorm_49565:'
blocks_b3_batchnorm_49567:+
blocks_b3_prelu_49570:
identity¢+blocks/b3/batchnorm/StatefulPartitionedCall¢&blocks/b3/conv/StatefulPartitionedCall¢'blocks/b3/prelu/StatefulPartitionedCallº
&blocks/b3/conv/StatefulPartitionedCallStatefulPartitionedCallinputsblocks_b3_conv_49538blocks_b3_conv_49540*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_blocks/b3/conv_layer_call_and_return_conditional_losses_495372(
&blocks/b3/conv/StatefulPartitionedCall¶
+blocks/b3/batchnorm/StatefulPartitionedCallStatefulPartitionedCall/blocks/b3/conv/StatefulPartitionedCall:output:0blocks_b3_batchnorm_49561blocks_b3_batchnorm_49563blocks_b3_batchnorm_49565blocks_b3_batchnorm_49567*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_blocks/b3/batchnorm_layer_call_and_return_conditional_losses_495602-
+blocks/b3/batchnorm/StatefulPartitionedCallÔ
'blocks/b3/prelu/StatefulPartitionedCallStatefulPartitionedCall4blocks/b3/batchnorm/StatefulPartitionedCall:output:0blocks_b3_prelu_49570*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_blocks/b3/prelu_layer_call_and_return_conditional_losses_495002)
'blocks/b3/prelu/StatefulPartitionedCall£
!blocks/b3/maxpool/PartitionedCallPartitionedCall0blocks/b3/prelu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_blocks/b3/maxpool_layer_call_and_return_conditional_losses_495142#
!blocks/b3/maxpool/PartitionedCall
IdentityIdentity*blocks/b3/maxpool/PartitionedCall:output:0,^blocks/b3/batchnorm/StatefulPartitionedCall'^blocks/b3/conv/StatefulPartitionedCall(^blocks/b3/prelu/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : 2Z
+blocks/b3/batchnorm/StatefulPartitionedCall+blocks/b3/batchnorm/StatefulPartitionedCall2P
&blocks/b3/conv/StatefulPartitionedCall&blocks/b3/conv/StatefulPartitionedCall2R
'blocks/b3/prelu/StatefulPartitionedCall'blocks/b3/prelu/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ
ã
)__inference_simplecnn_layer_call_fn_50864
input_sp
unknown:``
	unknown_0:``#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:`#
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13: 

unknown_14:0$

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20: 

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_spunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*:
_read_only_resource_inputs
	
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_simplecnn_layer_call_and_return_conditional_losses_507282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
"
_user_specified_name
input_sp
Ð
£
.__inference_blocks/b2/conv_layer_call_fn_52425

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_blocks/b2/conv_layer_call_and_return_conditional_losses_491292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ00: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
 
_user_specified_nameinputs
Â

A__inference_blocks_layer_call_and_return_conditional_losses_50106
blocks_input)
blocks_b1_50060:
blocks_b1_50062:
blocks_b1_50064:
blocks_b1_50066:
blocks_b1_50068:
blocks_b1_50070:%
blocks_b1_50072:`)
blocks_b2_50075:
blocks_b2_50077:
blocks_b2_50079:
blocks_b2_50081:
blocks_b2_50083:
blocks_b2_50085:%
blocks_b2_50087:0)
blocks_b3_50090:
blocks_b3_50092:
blocks_b3_50094:
blocks_b3_50096:
blocks_b3_50098:
blocks_b3_50100:%
blocks_b3_50102:
identity¢!blocks/b1/StatefulPartitionedCall¢!blocks/b2/StatefulPartitionedCall¢!blocks/b3/StatefulPartitionedCall
!blocks/b1/StatefulPartitionedCallStatefulPartitionedCallblocks_inputblocks_b1_50060blocks_b1_50062blocks_b1_50064blocks_b1_50066blocks_b1_50068blocks_b1_50070blocks_b1_50072*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_blocks/b1_layer_call_and_return_conditional_losses_487592#
!blocks/b1/StatefulPartitionedCall¤
!blocks/b2/StatefulPartitionedCallStatefulPartitionedCall*blocks/b1/StatefulPartitionedCall:output:0blocks_b2_50075blocks_b2_50077blocks_b2_50079blocks_b2_50081blocks_b2_50083blocks_b2_50085blocks_b2_50087*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_blocks/b2_layer_call_and_return_conditional_losses_491672#
!blocks/b2/StatefulPartitionedCall¤
!blocks/b3/StatefulPartitionedCallStatefulPartitionedCall*blocks/b2/StatefulPartitionedCall:output:0blocks_b3_50090blocks_b3_50092blocks_b3_50094blocks_b3_50096blocks_b3_50098blocks_b3_50100blocks_b3_50102*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_blocks/b3_layer_call_and_return_conditional_losses_495752#
!blocks/b3/StatefulPartitionedCallò
IdentityIdentity*blocks/b3/StatefulPartitionedCall:output:0"^blocks/b1/StatefulPartitionedCall"^blocks/b2/StatefulPartitionedCall"^blocks/b3/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : 2F
!blocks/b1/StatefulPartitionedCall!blocks/b1/StatefulPartitionedCall2F
!blocks/b2/StatefulPartitionedCall!blocks/b2/StatefulPartitionedCall2F
!blocks/b3/StatefulPartitionedCall!blocks/b3/StatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
&
_user_specified_nameblocks/input
µ


I__inference_blocks/b2/conv_layer_call_and_return_conditional_losses_49129

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ00: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
 
_user_specified_nameinputs


©
J__inference_blocks/b1/prelu_layer_call_and_return_conditional_losses_48684

inputs-
readvariableop_resource:`
identity¢ReadVariableOp_
ReluReluinputs*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ`ÿÿÿÿÿÿÿÿÿ2
Relu|
ReadVariableOpReadVariableOpreadvariableop_resource*"
_output_shapes
:`*
dtype02
ReadVariableOpV
NegNegReadVariableOp:value:0*
T0*"
_output_shapes
:`2
Neg`
Neg_1Neginputs*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ`ÿÿÿÿÿÿÿÿÿ2
Neg_1f
Relu_1Relu	Neg_1:y:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ`ÿÿÿÿÿÿÿÿÿ2
Relu_1s
mulMulNeg:y:0Relu_1:activations:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ`ÿÿÿÿÿÿÿÿÿ2
muls
addAddV2Relu:activations:0mul:z:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ`ÿÿÿÿÿÿÿÿÿ2
add}
IdentityIdentityadd:z:0^ReadVariableOp*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ`ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:"ÿÿÿÿÿÿÿÿÿ`ÿÿÿÿÿÿÿÿÿ: 2 
ReadVariableOpReadVariableOp:` \
8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ`ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
½
N__inference_blocks/b3/batchnorm_layer_call_and_return_conditional_losses_52666

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


©
J__inference_blocks/b2/prelu_layer_call_and_return_conditional_losses_49092

inputs-
readvariableop_resource:0
identity¢ReadVariableOp_
ReluReluinputs*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ0ÿÿÿÿÿÿÿÿÿ2
Relu|
ReadVariableOpReadVariableOpreadvariableop_resource*"
_output_shapes
:0*
dtype02
ReadVariableOpV
NegNegReadVariableOp:value:0*
T0*"
_output_shapes
:02
Neg`
Neg_1Neginputs*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ0ÿÿÿÿÿÿÿÿÿ2
Neg_1f
Relu_1Relu	Neg_1:y:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ0ÿÿÿÿÿÿÿÿÿ2
Relu_1s
mulMulNeg:y:0Relu_1:activations:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ0ÿÿÿÿÿÿÿÿÿ2
muls
addAddV2Relu:activations:0mul:z:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ0ÿÿÿÿÿÿÿÿÿ2
add}
IdentityIdentityadd:z:0^ReadVariableOp*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ0ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:"ÿÿÿÿÿÿÿÿÿ0ÿÿÿÿÿÿÿÿÿ: 2 
ReadVariableOpReadVariableOp:` \
8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ0ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë)
à
G__inference_fc_batchnorm_layer_call_and_return_conditional_losses_51923

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¤
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decayª
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡
D
(__inference_restored_function_body_48542

inputs
identity¨
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_probability_layer_call_and_return_conditional_losses_432772
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Å
G
+__inference_probability_layer_call_fn_42553

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_probability_layer_call_and_return_conditional_losses_425482
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à
b
F__inference_probability_layer_call_and_return_conditional_losses_42548

inputs
identityW
SigmoidSigmoidinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


N__inference_blocks/b1/batchnorm_layer_call_and_return_conditional_losses_48567

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
d
H__inference_globalavgpool_layer_call_and_return_conditional_losses_50162

inputs
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
M
1__inference_blocks/b1/maxpool_layer_call_fn_48704

inputs
identityð
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_blocks/b1/maxpool_layer_call_and_return_conditional_losses_486982
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ


I__inference_blocks/b1/conv_layer_call_and_return_conditional_losses_52292

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ``: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
ô
½
N__inference_blocks/b3/batchnorm_layer_call_and_return_conditional_losses_52702

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1þ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À

N__inference_blocks/b1/batchnorm_layer_call_and_return_conditional_losses_48744

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ``:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ú
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ``: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
ä
Î
3__inference_blocks/b1/batchnorm_layer_call_fn_52318

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_blocks/b1/batchnorm_layer_call_and_return_conditional_losses_486112
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê

/__inference_blocks/b3/prelu_layer_call_fn_49508

inputs
unknown:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_blocks/b3/prelu_layer_call_and_return_conditional_losses_495002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:` \
8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ºB

D__inference_simplecnn_layer_call_and_return_conditional_losses_50469

inputsA
/featurewise_std_reshape_readvariableop_resource:``C
1featurewise_std_reshape_1_readvariableop_resource:``&
blocks_50372:
blocks_50374:
blocks_50376:
blocks_50378:
blocks_50380:
blocks_50382:"
blocks_50384:`&
blocks_50386:
blocks_50388:
blocks_50390:
blocks_50392:
blocks_50394:
blocks_50396:"
blocks_50398:0&
blocks_50400:
blocks_50402:
blocks_50404:
blocks_50406:
blocks_50408:
blocks_50410:"
blocks_50412:
dense_50427:
dense_50429: 
fc_batchnorm_50432: 
fc_batchnorm_50434: 
fc_batchnorm_50436: 
fc_batchnorm_50438:
fc_prelu_50448:!
dense_out_50462:
dense_out_50464:
identity¢blocks/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢!dense_out/StatefulPartitionedCall¢$fc_batchnorm/StatefulPartitionedCall¢ fc_prelu/StatefulPartitionedCall¢&featurewise_std/Reshape/ReadVariableOp¢(featurewise_std/Reshape_1/ReadVariableOpÀ
&featurewise_std/Reshape/ReadVariableOpReadVariableOp/featurewise_std_reshape_readvariableop_resource*
_output_shapes

:``*
dtype02(
&featurewise_std/Reshape/ReadVariableOp
featurewise_std/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   `   `   2
featurewise_std/Reshape/shapeÂ
featurewise_std/ReshapeReshape.featurewise_std/Reshape/ReadVariableOp:value:0&featurewise_std/Reshape/shape:output:0*
T0*"
_output_shapes
:``2
featurewise_std/ReshapeÆ
(featurewise_std/Reshape_1/ReadVariableOpReadVariableOp1featurewise_std_reshape_1_readvariableop_resource*
_output_shapes

:``*
dtype02*
(featurewise_std/Reshape_1/ReadVariableOp
featurewise_std/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   `   `   2!
featurewise_std/Reshape_1/shapeÊ
featurewise_std/Reshape_1Reshape0featurewise_std/Reshape_1/ReadVariableOp:value:0(featurewise_std/Reshape_1/shape:output:0*
T0*"
_output_shapes
:``2
featurewise_std/Reshape_1
featurewise_std/subSubinputs featurewise_std/Reshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
featurewise_std/sub
featurewise_std/SqrtSqrt"featurewise_std/Reshape_1:output:0*
T0*"
_output_shapes
:``2
featurewise_std/Sqrt{
featurewise_std/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32
featurewise_std/Maximum/y¨
featurewise_std/MaximumMaximumfeaturewise_std/Sqrt:y:0"featurewise_std/Maximum/y:output:0*
T0*"
_output_shapes
:``2
featurewise_std/Maximum©
featurewise_std/truedivRealDivfeaturewise_std/sub:z:0featurewise_std/Maximum:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
featurewise_std/truedivî
#insert_channel_axis/PartitionedCallPartitionedCallfeaturewise_std/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_484102%
#insert_channel_axis/PartitionedCallè
blocks/StatefulPartitionedCallStatefulPartitionedCall,insert_channel_axis/PartitionedCall:output:0blocks_50372blocks_50374blocks_50376blocks_50378blocks_50380blocks_50382blocks_50384blocks_50386blocks_50388blocks_50390blocks_50392blocks_50394blocks_50396blocks_50398blocks_50400blocks_50402blocks_50404blocks_50406blocks_50408blocks_50410blocks_50412*!
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*7
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_blocks_layer_call_and_return_conditional_losses_498222 
blocks/StatefulPartitionedCall
globalavgpool/PartitionedCallPartitionedCall'blocks/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_globalavgpool_layer_call_and_return_conditional_losses_501622
globalavgpool/PartitionedCall¥
dense/StatefulPartitionedCallStatefulPartitionedCall&globalavgpool/PartitionedCall:output:0dense_50427dense_50429*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_504262
dense/StatefulPartitionedCallô
$fc_batchnorm/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0fc_batchnorm_50432fc_batchnorm_50434fc_batchnorm_50436fc_batchnorm_50438*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_fc_batchnorm_layer_call_and_return_conditional_losses_501922&
$fc_batchnorm/StatefulPartitionedCall
fc_dropout/PartitionedCallPartitionedCall-fc_batchnorm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_fc_dropout_layer_call_and_return_conditional_losses_504462
fc_dropout/PartitionedCall
 fc_prelu/StatefulPartitionedCallStatefulPartitionedCall#fc_dropout/PartitionedCall:output:0fc_prelu_50448*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_fc_prelu_layer_call_and_return_conditional_losses_503432"
 fc_prelu/StatefulPartitionedCall¼
!dense_out/StatefulPartitionedCallStatefulPartitionedCall)fc_prelu/StatefulPartitionedCall:output:0dense_out_50462dense_out_50464*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_out_layer_call_and_return_conditional_losses_504612#
!dense_out/StatefulPartitionedCallå
probability/PartitionedCallPartitionedCall*dense_out/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_485422
probability/PartitionedCallû
IdentityIdentity$probability/PartitionedCall:output:0^blocks/StatefulPartitionedCall^dense/StatefulPartitionedCall"^dense_out/StatefulPartitionedCall%^fc_batchnorm/StatefulPartitionedCall!^fc_prelu/StatefulPartitionedCall'^featurewise_std/Reshape/ReadVariableOp)^featurewise_std/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
blocks/StatefulPartitionedCallblocks/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!dense_out/StatefulPartitionedCall!dense_out/StatefulPartitionedCall2L
$fc_batchnorm/StatefulPartitionedCall$fc_batchnorm/StatefulPartitionedCall2D
 fc_prelu/StatefulPartitionedCall fc_prelu/StatefulPartitionedCall2P
&featurewise_std/Reshape/ReadVariableOp&featurewise_std/Reshape/ReadVariableOp2T
(featurewise_std/Reshape_1/ReadVariableOp(featurewise_std/Reshape_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
À

N__inference_blocks/b1/batchnorm_layer_call_and_return_conditional_losses_52398

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ``:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ú
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ``: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
Ü
M
1__inference_blocks/b2/maxpool_layer_call_fn_49112

inputs
identityð
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_blocks/b2/maxpool_layer_call_and_return_conditional_losses_491062
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


D__inference_blocks/b3_layer_call_and_return_conditional_losses_49747
blocks_b3_conv_input.
blocks_b3_conv_49728:"
blocks_b3_conv_49730:'
blocks_b3_batchnorm_49733:'
blocks_b3_batchnorm_49735:'
blocks_b3_batchnorm_49737:'
blocks_b3_batchnorm_49739:+
blocks_b3_prelu_49742:
identity¢+blocks/b3/batchnorm/StatefulPartitionedCall¢&blocks/b3/conv/StatefulPartitionedCall¢'blocks/b3/prelu/StatefulPartitionedCallÈ
&blocks/b3/conv/StatefulPartitionedCallStatefulPartitionedCallblocks_b3_conv_inputblocks_b3_conv_49728blocks_b3_conv_49730*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_blocks/b3/conv_layer_call_and_return_conditional_losses_495372(
&blocks/b3/conv/StatefulPartitionedCall¶
+blocks/b3/batchnorm/StatefulPartitionedCallStatefulPartitionedCall/blocks/b3/conv/StatefulPartitionedCall:output:0blocks_b3_batchnorm_49733blocks_b3_batchnorm_49735blocks_b3_batchnorm_49737blocks_b3_batchnorm_49739*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_blocks/b3/batchnorm_layer_call_and_return_conditional_losses_495602-
+blocks/b3/batchnorm/StatefulPartitionedCallÔ
'blocks/b3/prelu/StatefulPartitionedCallStatefulPartitionedCall4blocks/b3/batchnorm/StatefulPartitionedCall:output:0blocks_b3_prelu_49742*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_blocks/b3/prelu_layer_call_and_return_conditional_losses_495002)
'blocks/b3/prelu/StatefulPartitionedCall£
!blocks/b3/maxpool/PartitionedCallPartitionedCall0blocks/b3/prelu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_blocks/b3/maxpool_layer_call_and_return_conditional_losses_495142#
!blocks/b3/maxpool/PartitionedCall
IdentityIdentity*blocks/b3/maxpool/PartitionedCall:output:0,^blocks/b3/batchnorm/StatefulPartitionedCall'^blocks/b3/conv/StatefulPartitionedCall(^blocks/b3/prelu/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : 2Z
+blocks/b3/batchnorm/StatefulPartitionedCall+blocks/b3/batchnorm/StatefulPartitionedCall2P
&blocks/b3/conv/StatefulPartitionedCall&blocks/b3/conv/StatefulPartitionedCall2R
'blocks/b3/prelu/StatefulPartitionedCall'blocks/b3/prelu/StatefulPartitionedCall:e a
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.
_user_specified_nameblocks/b3/conv_input
æ
Î
3__inference_blocks/b2/batchnorm_layer_call_fn_52448

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_blocks/b2/batchnorm_layer_call_and_return_conditional_losses_489752
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


N__inference_blocks/b3/batchnorm_layer_call_and_return_conditional_losses_49383

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿	

C__inference_fc_prelu_layer_call_and_return_conditional_losses_50343

inputs%
readvariableop_resource:
identity¢ReadVariableOpW
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relut
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpN
NegNegReadVariableOp:value:0*
T0*
_output_shapes
:2
NegX
Neg_1Neginputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Neg_1^
Relu_1Relu	Neg_1:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu_1k
mulMulNeg:y:0Relu_1:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
mulk
addAddV2Relu:activations:0mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
addu
IdentityIdentityadd:z:0^ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 2 
ReadVariableOpReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


©
J__inference_blocks/b3/prelu_layer_call_and_return_conditional_losses_49500

inputs-
readvariableop_resource:
identity¢ReadVariableOp_
ReluReluinputs*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu|
ReadVariableOpReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOpV
NegNegReadVariableOp:value:0*
T0*"
_output_shapes
:2
Neg`
Neg_1Neginputs*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Neg_1f
Relu_1Relu	Neg_1:y:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu_1s
mulMulNeg:y:0Relu_1:activations:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
muls
addAddV2Relu:activations:0mul:z:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
add}
IdentityIdentityadd:z:0^ReadVariableOp*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 2 
ReadVariableOpReadVariableOp:` \
8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¶
&__inference_blocks_layer_call_fn_49867
blocks_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:`#
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11: 

unknown_12:0$

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18: 

unknown_19:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallblocks_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19*!
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*7
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_blocks_layer_call_and_return_conditional_losses_498222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
&
_user_specified_nameblocks/input

Î
3__inference_blocks/b2/batchnorm_layer_call_fn_52487

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_blocks/b2/batchnorm_layer_call_and_return_conditional_losses_492172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ00: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
 
_user_specified_nameinputs
	
£
)__inference_blocks/b2_layer_call_fn_52084

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:0
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_blocks/b2_layer_call_and_return_conditional_losses_491672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ00: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
 
_user_specified_nameinputs
¼
½
N__inference_blocks/b2/batchnorm_layer_call_and_return_conditional_losses_49019

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä
Î
3__inference_blocks/b3/batchnorm_layer_call_fn_52604

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_blocks/b3/batchnorm_layer_call_and_return_conditional_losses_494272
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Q
Ä
__inference__traced_save_52848
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop1
-savev2_fc_batchnorm_gamma_read_readvariableop0
,savev2_fc_batchnorm_beta_read_readvariableop7
3savev2_fc_batchnorm_moving_mean_read_readvariableop;
7savev2_fc_batchnorm_moving_variance_read_readvariableop-
)savev2_fc_prelu_alpha_read_readvariableop/
+savev2_dense_out_kernel_read_readvariableop-
)savev2_dense_out_bias_read_readvariableop4
0savev2_blocks_b1_conv_kernel_read_readvariableop2
.savev2_blocks_b1_conv_bias_read_readvariableop8
4savev2_blocks_b1_batchnorm_gamma_read_readvariableop7
3savev2_blocks_b1_batchnorm_beta_read_readvariableop>
:savev2_blocks_b1_batchnorm_moving_mean_read_readvariableopB
>savev2_blocks_b1_batchnorm_moving_variance_read_readvariableop4
0savev2_blocks_b1_prelu_alpha_read_readvariableop4
0savev2_blocks_b2_conv_kernel_read_readvariableop2
.savev2_blocks_b2_conv_bias_read_readvariableop8
4savev2_blocks_b2_batchnorm_gamma_read_readvariableop7
3savev2_blocks_b2_batchnorm_beta_read_readvariableop>
:savev2_blocks_b2_batchnorm_moving_mean_read_readvariableopB
>savev2_blocks_b2_batchnorm_moving_variance_read_readvariableop4
0savev2_blocks_b2_prelu_alpha_read_readvariableop4
0savev2_blocks_b3_conv_kernel_read_readvariableop2
.savev2_blocks_b3_conv_bias_read_readvariableop8
4savev2_blocks_b3_batchnorm_gamma_read_readvariableop7
3savev2_blocks_b3_batchnorm_beta_read_readvariableop>
:savev2_blocks_b3_batchnorm_moving_mean_read_readvariableopB
>savev2_blocks_b3_batchnorm_moving_variance_read_readvariableop4
0savev2_blocks_b3_prelu_alpha_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_2_read_readvariableop-
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
ShardedFilenameò
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*
valueúB÷*B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/alpha/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÜ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop-savev2_fc_batchnorm_gamma_read_readvariableop,savev2_fc_batchnorm_beta_read_readvariableop3savev2_fc_batchnorm_moving_mean_read_readvariableop7savev2_fc_batchnorm_moving_variance_read_readvariableop)savev2_fc_prelu_alpha_read_readvariableop+savev2_dense_out_kernel_read_readvariableop)savev2_dense_out_bias_read_readvariableop0savev2_blocks_b1_conv_kernel_read_readvariableop.savev2_blocks_b1_conv_bias_read_readvariableop4savev2_blocks_b1_batchnorm_gamma_read_readvariableop3savev2_blocks_b1_batchnorm_beta_read_readvariableop:savev2_blocks_b1_batchnorm_moving_mean_read_readvariableop>savev2_blocks_b1_batchnorm_moving_variance_read_readvariableop0savev2_blocks_b1_prelu_alpha_read_readvariableop0savev2_blocks_b2_conv_kernel_read_readvariableop.savev2_blocks_b2_conv_bias_read_readvariableop4savev2_blocks_b2_batchnorm_gamma_read_readvariableop3savev2_blocks_b2_batchnorm_beta_read_readvariableop:savev2_blocks_b2_batchnorm_moving_mean_read_readvariableop>savev2_blocks_b2_batchnorm_moving_variance_read_readvariableop0savev2_blocks_b2_prelu_alpha_read_readvariableop0savev2_blocks_b3_conv_kernel_read_readvariableop.savev2_blocks_b3_conv_bias_read_readvariableop4savev2_blocks_b3_batchnorm_gamma_read_readvariableop3savev2_blocks_b3_batchnorm_beta_read_readvariableop:savev2_blocks_b3_batchnorm_moving_mean_read_readvariableop>savev2_blocks_b3_batchnorm_moving_variance_read_readvariableop0savev2_blocks_b3_prelu_alpha_read_readvariableop savev2_total_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_2_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_negatives_read_readvariableop+savev2_true_positives_1_read_readvariableop*savev2_false_positives_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *8
dtypes.
,2*	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*Ç
_input_shapesµ
²: :``:``: ::::::::::::::::`:::::::0:::::::: : : : ::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:``:$ 

_output_shapes

:``:

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:`:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:0:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
::(!$
"
_output_shapes
::"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: : &

_output_shapes
:: '

_output_shapes
:: (

_output_shapes
:: )

_output_shapes
::*

_output_shapes
: 
í
å 
D__inference_simplecnn_layer_call_and_return_conditional_losses_51390

inputsA
/featurewise_std_reshape_readvariableop_resource:``C
1featurewise_std_reshape_1_readvariableop_resource:``X
>blocks_blocks_b1_blocks_b1_conv_conv2d_readvariableop_resource:M
?blocks_blocks_b1_blocks_b1_conv_biasadd_readvariableop_resource:J
<blocks_blocks_b1_blocks_b1_batchnorm_readvariableop_resource:L
>blocks_blocks_b1_blocks_b1_batchnorm_readvariableop_1_resource:[
Mblocks_blocks_b1_blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_resource:]
Oblocks_blocks_b1_blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_1_resource:N
8blocks_blocks_b1_blocks_b1_prelu_readvariableop_resource:`X
>blocks_blocks_b2_blocks_b2_conv_conv2d_readvariableop_resource:M
?blocks_blocks_b2_blocks_b2_conv_biasadd_readvariableop_resource:J
<blocks_blocks_b2_blocks_b2_batchnorm_readvariableop_resource:L
>blocks_blocks_b2_blocks_b2_batchnorm_readvariableop_1_resource:[
Mblocks_blocks_b2_blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_resource:]
Oblocks_blocks_b2_blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_1_resource:N
8blocks_blocks_b2_blocks_b2_prelu_readvariableop_resource:0X
>blocks_blocks_b3_blocks_b3_conv_conv2d_readvariableop_resource:M
?blocks_blocks_b3_blocks_b3_conv_biasadd_readvariableop_resource:J
<blocks_blocks_b3_blocks_b3_batchnorm_readvariableop_resource:L
>blocks_blocks_b3_blocks_b3_batchnorm_readvariableop_1_resource:[
Mblocks_blocks_b3_blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_resource:]
Oblocks_blocks_b3_blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_1_resource:N
8blocks_blocks_b3_blocks_b3_prelu_readvariableop_resource:6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:<
.fc_batchnorm_batchnorm_readvariableop_resource:@
2fc_batchnorm_batchnorm_mul_readvariableop_resource:>
0fc_batchnorm_batchnorm_readvariableop_1_resource:>
0fc_batchnorm_batchnorm_readvariableop_2_resource:.
 fc_prelu_readvariableop_resource::
(dense_out_matmul_readvariableop_resource:7
)dense_out_biasadd_readvariableop_resource:
identity¢Dblocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp¢Fblocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1¢3blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp¢5blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp_1¢6blocks/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp¢5blocks/blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOp¢/blocks/blocks/b1/blocks/b1/prelu/ReadVariableOp¢Dblocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp¢Fblocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1¢3blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp¢5blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp_1¢6blocks/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp¢5blocks/blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp¢/blocks/blocks/b2/blocks/b2/prelu/ReadVariableOp¢Dblocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp¢Fblocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1¢3blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp¢5blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp_1¢6blocks/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp¢5blocks/blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp¢/blocks/blocks/b3/blocks/b3/prelu/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢ dense_out/BiasAdd/ReadVariableOp¢dense_out/MatMul/ReadVariableOp¢%fc_batchnorm/batchnorm/ReadVariableOp¢'fc_batchnorm/batchnorm/ReadVariableOp_1¢'fc_batchnorm/batchnorm/ReadVariableOp_2¢)fc_batchnorm/batchnorm/mul/ReadVariableOp¢fc_prelu/ReadVariableOp¢&featurewise_std/Reshape/ReadVariableOp¢(featurewise_std/Reshape_1/ReadVariableOpÀ
&featurewise_std/Reshape/ReadVariableOpReadVariableOp/featurewise_std_reshape_readvariableop_resource*
_output_shapes

:``*
dtype02(
&featurewise_std/Reshape/ReadVariableOp
featurewise_std/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   `   `   2
featurewise_std/Reshape/shapeÂ
featurewise_std/ReshapeReshape.featurewise_std/Reshape/ReadVariableOp:value:0&featurewise_std/Reshape/shape:output:0*
T0*"
_output_shapes
:``2
featurewise_std/ReshapeÆ
(featurewise_std/Reshape_1/ReadVariableOpReadVariableOp1featurewise_std_reshape_1_readvariableop_resource*
_output_shapes

:``*
dtype02*
(featurewise_std/Reshape_1/ReadVariableOp
featurewise_std/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   `   `   2!
featurewise_std/Reshape_1/shapeÊ
featurewise_std/Reshape_1Reshape0featurewise_std/Reshape_1/ReadVariableOp:value:0(featurewise_std/Reshape_1/shape:output:0*
T0*"
_output_shapes
:``2
featurewise_std/Reshape_1
featurewise_std/subSubinputs featurewise_std/Reshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
featurewise_std/sub
featurewise_std/SqrtSqrt"featurewise_std/Reshape_1:output:0*
T0*"
_output_shapes
:``2
featurewise_std/Sqrt{
featurewise_std/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32
featurewise_std/Maximum/y¨
featurewise_std/MaximumMaximumfeaturewise_std/Sqrt:y:0"featurewise_std/Maximum/y:output:0*
T0*"
_output_shapes
:``2
featurewise_std/Maximum©
featurewise_std/truedivRealDivfeaturewise_std/sub:z:0featurewise_std/Maximum:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
featurewise_std/truedivî
#insert_channel_axis/PartitionedCallPartitionedCallfeaturewise_std/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_484102%
#insert_channel_axis/PartitionedCallõ
5blocks/blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOpReadVariableOp>blocks_blocks_b1_blocks_b1_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype027
5blocks/blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOp©
&blocks/blocks/b1/blocks/b1/conv/Conv2DConv2D,insert_channel_axis/PartitionedCall:output:0=blocks/blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*
paddingSAME*
strides
2(
&blocks/blocks/b1/blocks/b1/conv/Conv2Dì
6blocks/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOpReadVariableOp?blocks_blocks_b1_blocks_b1_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6blocks/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp
'blocks/blocks/b1/blocks/b1/conv/BiasAddBiasAdd/blocks/blocks/b1/blocks/b1/conv/Conv2D:output:0>blocks/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2)
'blocks/blocks/b1/blocks/b1/conv/BiasAddã
3blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOpReadVariableOp<blocks_blocks_b1_blocks_b1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype025
3blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOpé
5blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp_1ReadVariableOp>blocks_blocks_b1_blocks_b1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype027
5blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp_1
Dblocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOpReadVariableOpMblocks_blocks_b1_blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dblocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp
Fblocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOblocks_blocks_b1_blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Fblocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1Ò
5blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3FusedBatchNormV30blocks/blocks/b1/blocks/b1/conv/BiasAdd:output:0;blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp:value:0=blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp_1:value:0Lblocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp:value:0Nblocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ``:::::*
epsilon%o:*
is_training( 27
5blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3Ë
%blocks/blocks/b1/blocks/b1/prelu/ReluRelu9blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2'
%blocks/blocks/b1/blocks/b1/prelu/Reluß
/blocks/blocks/b1/blocks/b1/prelu/ReadVariableOpReadVariableOp8blocks_blocks_b1_blocks_b1_prelu_readvariableop_resource*"
_output_shapes
:`*
dtype021
/blocks/blocks/b1/blocks/b1/prelu/ReadVariableOp¹
$blocks/blocks/b1/blocks/b1/prelu/NegNeg7blocks/blocks/b1/blocks/b1/prelu/ReadVariableOp:value:0*
T0*"
_output_shapes
:`2&
$blocks/blocks/b1/blocks/b1/prelu/NegÌ
&blocks/blocks/b1/blocks/b1/prelu/Neg_1Neg9blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2(
&blocks/blocks/b1/blocks/b1/prelu/Neg_1À
'blocks/blocks/b1/blocks/b1/prelu/Relu_1Relu*blocks/blocks/b1/blocks/b1/prelu/Neg_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2)
'blocks/blocks/b1/blocks/b1/prelu/Relu_1î
$blocks/blocks/b1/blocks/b1/prelu/mulMul(blocks/blocks/b1/blocks/b1/prelu/Neg:y:05blocks/blocks/b1/blocks/b1/prelu/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2&
$blocks/blocks/b1/blocks/b1/prelu/mulî
$blocks/blocks/b1/blocks/b1/prelu/addAddV23blocks/blocks/b1/blocks/b1/prelu/Relu:activations:0(blocks/blocks/b1/blocks/b1/prelu/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2&
$blocks/blocks/b1/blocks/b1/prelu/addù
*blocks/blocks/b1/blocks/b1/maxpool/MaxPoolMaxPool(blocks/blocks/b1/blocks/b1/prelu/add:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*
ksize
*
paddingSAME*
strides
2,
*blocks/blocks/b1/blocks/b1/maxpool/MaxPoolõ
5blocks/blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOpReadVariableOp>blocks_blocks_b2_blocks_b2_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype027
5blocks/blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp°
&blocks/blocks/b2/blocks/b2/conv/Conv2DConv2D3blocks/blocks/b1/blocks/b1/maxpool/MaxPool:output:0=blocks/blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*
paddingSAME*
strides
2(
&blocks/blocks/b2/blocks/b2/conv/Conv2Dì
6blocks/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOpReadVariableOp?blocks_blocks_b2_blocks_b2_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6blocks/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp
'blocks/blocks/b2/blocks/b2/conv/BiasAddBiasAdd/blocks/blocks/b2/blocks/b2/conv/Conv2D:output:0>blocks/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002)
'blocks/blocks/b2/blocks/b2/conv/BiasAddã
3blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOpReadVariableOp<blocks_blocks_b2_blocks_b2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype025
3blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOpé
5blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp_1ReadVariableOp>blocks_blocks_b2_blocks_b2_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype027
5blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp_1
Dblocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOpReadVariableOpMblocks_blocks_b2_blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dblocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp
Fblocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOblocks_blocks_b2_blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Fblocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1Ò
5blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3FusedBatchNormV30blocks/blocks/b2/blocks/b2/conv/BiasAdd:output:0;blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp:value:0=blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp_1:value:0Lblocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp:value:0Nblocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00:::::*
epsilon%o:*
is_training( 27
5blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3Ë
%blocks/blocks/b2/blocks/b2/prelu/ReluRelu9blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002'
%blocks/blocks/b2/blocks/b2/prelu/Reluß
/blocks/blocks/b2/blocks/b2/prelu/ReadVariableOpReadVariableOp8blocks_blocks_b2_blocks_b2_prelu_readvariableop_resource*"
_output_shapes
:0*
dtype021
/blocks/blocks/b2/blocks/b2/prelu/ReadVariableOp¹
$blocks/blocks/b2/blocks/b2/prelu/NegNeg7blocks/blocks/b2/blocks/b2/prelu/ReadVariableOp:value:0*
T0*"
_output_shapes
:02&
$blocks/blocks/b2/blocks/b2/prelu/NegÌ
&blocks/blocks/b2/blocks/b2/prelu/Neg_1Neg9blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002(
&blocks/blocks/b2/blocks/b2/prelu/Neg_1À
'blocks/blocks/b2/blocks/b2/prelu/Relu_1Relu*blocks/blocks/b2/blocks/b2/prelu/Neg_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002)
'blocks/blocks/b2/blocks/b2/prelu/Relu_1î
$blocks/blocks/b2/blocks/b2/prelu/mulMul(blocks/blocks/b2/blocks/b2/prelu/Neg:y:05blocks/blocks/b2/blocks/b2/prelu/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002&
$blocks/blocks/b2/blocks/b2/prelu/mulî
$blocks/blocks/b2/blocks/b2/prelu/addAddV23blocks/blocks/b2/blocks/b2/prelu/Relu:activations:0(blocks/blocks/b2/blocks/b2/prelu/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002&
$blocks/blocks/b2/blocks/b2/prelu/addù
*blocks/blocks/b2/blocks/b2/maxpool/MaxPoolMaxPool(blocks/blocks/b2/blocks/b2/prelu/add:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2,
*blocks/blocks/b2/blocks/b2/maxpool/MaxPoolõ
5blocks/blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOpReadVariableOp>blocks_blocks_b3_blocks_b3_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype027
5blocks/blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp°
&blocks/blocks/b3/blocks/b3/conv/Conv2DConv2D3blocks/blocks/b2/blocks/b2/maxpool/MaxPool:output:0=blocks/blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2(
&blocks/blocks/b3/blocks/b3/conv/Conv2Dì
6blocks/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOpReadVariableOp?blocks_blocks_b3_blocks_b3_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6blocks/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp
'blocks/blocks/b3/blocks/b3/conv/BiasAddBiasAdd/blocks/blocks/b3/blocks/b3/conv/Conv2D:output:0>blocks/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'blocks/blocks/b3/blocks/b3/conv/BiasAddã
3blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOpReadVariableOp<blocks_blocks_b3_blocks_b3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype025
3blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOpé
5blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp_1ReadVariableOp>blocks_blocks_b3_blocks_b3_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype027
5blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp_1
Dblocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOpReadVariableOpMblocks_blocks_b3_blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dblocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp
Fblocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOblocks_blocks_b3_blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Fblocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1Ò
5blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3FusedBatchNormV30blocks/blocks/b3/blocks/b3/conv/BiasAdd:output:0;blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp:value:0=blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp_1:value:0Lblocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp:value:0Nblocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 27
5blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3Ë
%blocks/blocks/b3/blocks/b3/prelu/ReluRelu9blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%blocks/blocks/b3/blocks/b3/prelu/Reluß
/blocks/blocks/b3/blocks/b3/prelu/ReadVariableOpReadVariableOp8blocks_blocks_b3_blocks_b3_prelu_readvariableop_resource*"
_output_shapes
:*
dtype021
/blocks/blocks/b3/blocks/b3/prelu/ReadVariableOp¹
$blocks/blocks/b3/blocks/b3/prelu/NegNeg7blocks/blocks/b3/blocks/b3/prelu/ReadVariableOp:value:0*
T0*"
_output_shapes
:2&
$blocks/blocks/b3/blocks/b3/prelu/NegÌ
&blocks/blocks/b3/blocks/b3/prelu/Neg_1Neg9blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&blocks/blocks/b3/blocks/b3/prelu/Neg_1À
'blocks/blocks/b3/blocks/b3/prelu/Relu_1Relu*blocks/blocks/b3/blocks/b3/prelu/Neg_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'blocks/blocks/b3/blocks/b3/prelu/Relu_1î
$blocks/blocks/b3/blocks/b3/prelu/mulMul(blocks/blocks/b3/blocks/b3/prelu/Neg:y:05blocks/blocks/b3/blocks/b3/prelu/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$blocks/blocks/b3/blocks/b3/prelu/mulî
$blocks/blocks/b3/blocks/b3/prelu/addAddV23blocks/blocks/b3/blocks/b3/prelu/Relu:activations:0(blocks/blocks/b3/blocks/b3/prelu/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$blocks/blocks/b3/blocks/b3/prelu/addù
*blocks/blocks/b3/blocks/b3/maxpool/MaxPoolMaxPool(blocks/blocks/b3/blocks/b3/prelu/add:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2,
*blocks/blocks/b3/blocks/b3/maxpool/MaxPool
$globalavgpool/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2&
$globalavgpool/Mean/reduction_indicesÆ
globalavgpool/MeanMean3blocks/blocks/b3/blocks/b3/maxpool/MaxPool:output:0-globalavgpool/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
globalavgpool/Mean
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulglobalavgpool/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAdd¹
%fc_batchnorm/batchnorm/ReadVariableOpReadVariableOp.fc_batchnorm_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02'
%fc_batchnorm/batchnorm/ReadVariableOp
fc_batchnorm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
fc_batchnorm/batchnorm/add/y¼
fc_batchnorm/batchnorm/addAddV2-fc_batchnorm/batchnorm/ReadVariableOp:value:0%fc_batchnorm/batchnorm/add/y:output:0*
T0*
_output_shapes
:2
fc_batchnorm/batchnorm/add
fc_batchnorm/batchnorm/RsqrtRsqrtfc_batchnorm/batchnorm/add:z:0*
T0*
_output_shapes
:2
fc_batchnorm/batchnorm/RsqrtÅ
)fc_batchnorm/batchnorm/mul/ReadVariableOpReadVariableOp2fc_batchnorm_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02+
)fc_batchnorm/batchnorm/mul/ReadVariableOp¹
fc_batchnorm/batchnorm/mulMul fc_batchnorm/batchnorm/Rsqrt:y:01fc_batchnorm/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
fc_batchnorm/batchnorm/mul­
fc_batchnorm/batchnorm/mul_1Muldense/BiasAdd:output:0fc_batchnorm/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
fc_batchnorm/batchnorm/mul_1¿
'fc_batchnorm/batchnorm/ReadVariableOp_1ReadVariableOp0fc_batchnorm_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'fc_batchnorm/batchnorm/ReadVariableOp_1¹
fc_batchnorm/batchnorm/mul_2Mul/fc_batchnorm/batchnorm/ReadVariableOp_1:value:0fc_batchnorm/batchnorm/mul:z:0*
T0*
_output_shapes
:2
fc_batchnorm/batchnorm/mul_2¿
'fc_batchnorm/batchnorm/ReadVariableOp_2ReadVariableOp0fc_batchnorm_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02)
'fc_batchnorm/batchnorm/ReadVariableOp_2·
fc_batchnorm/batchnorm/subSub/fc_batchnorm/batchnorm/ReadVariableOp_2:value:0 fc_batchnorm/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
fc_batchnorm/batchnorm/sub¹
fc_batchnorm/batchnorm/add_1AddV2 fc_batchnorm/batchnorm/mul_1:z:0fc_batchnorm/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
fc_batchnorm/batchnorm/add_1
fc_dropout/IdentityIdentity fc_batchnorm/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
fc_dropout/Identityv
fc_prelu/ReluRelufc_dropout/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
fc_prelu/Relu
fc_prelu/ReadVariableOpReadVariableOp fc_prelu_readvariableop_resource*
_output_shapes
:*
dtype02
fc_prelu/ReadVariableOpi
fc_prelu/NegNegfc_prelu/ReadVariableOp:value:0*
T0*
_output_shapes
:2
fc_prelu/Negw
fc_prelu/Neg_1Negfc_dropout/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
fc_prelu/Neg_1p
fc_prelu/Relu_1Relufc_prelu/Neg_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
fc_prelu/Relu_1
fc_prelu/mulMulfc_prelu/Neg:y:0fc_prelu/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
fc_prelu/mul
fc_prelu/addAddV2fc_prelu/Relu:activations:0fc_prelu/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
fc_prelu/add«
dense_out/MatMul/ReadVariableOpReadVariableOp(dense_out_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
dense_out/MatMul/ReadVariableOp
dense_out/MatMulMatMulfc_prelu/add:z:0'dense_out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_out/MatMulª
 dense_out/BiasAdd/ReadVariableOpReadVariableOp)dense_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_out/BiasAdd/ReadVariableOp©
dense_out/BiasAddBiasAdddense_out/MatMul:product:0(dense_out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_out/BiasAddÕ
probability/PartitionedCallPartitionedCalldense_out/BiasAdd:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_485422
probability/PartitionedCalló
IdentityIdentity$probability/PartitionedCall:output:0E^blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOpG^blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_14^blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp6^blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp_17^blocks/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp6^blocks/blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOp0^blocks/blocks/b1/blocks/b1/prelu/ReadVariableOpE^blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOpG^blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_14^blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp6^blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp_17^blocks/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp6^blocks/blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp0^blocks/blocks/b2/blocks/b2/prelu/ReadVariableOpE^blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOpG^blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_14^blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp6^blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp_17^blocks/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp6^blocks/blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp0^blocks/blocks/b3/blocks/b3/prelu/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp!^dense_out/BiasAdd/ReadVariableOp ^dense_out/MatMul/ReadVariableOp&^fc_batchnorm/batchnorm/ReadVariableOp(^fc_batchnorm/batchnorm/ReadVariableOp_1(^fc_batchnorm/batchnorm/ReadVariableOp_2*^fc_batchnorm/batchnorm/mul/ReadVariableOp^fc_prelu/ReadVariableOp'^featurewise_std/Reshape/ReadVariableOp)^featurewise_std/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
Dblocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOpDblocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp2
Fblocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1Fblocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_12j
3blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp3blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp2n
5blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp_15blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp_12p
6blocks/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp6blocks/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp2n
5blocks/blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOp5blocks/blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOp2b
/blocks/blocks/b1/blocks/b1/prelu/ReadVariableOp/blocks/blocks/b1/blocks/b1/prelu/ReadVariableOp2
Dblocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOpDblocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp2
Fblocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1Fblocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_12j
3blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp3blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp2n
5blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp_15blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp_12p
6blocks/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp6blocks/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp2n
5blocks/blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp5blocks/blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp2b
/blocks/blocks/b2/blocks/b2/prelu/ReadVariableOp/blocks/blocks/b2/blocks/b2/prelu/ReadVariableOp2
Dblocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOpDblocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp2
Fblocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1Fblocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_12j
3blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp3blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp2n
5blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp_15blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp_12p
6blocks/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp6blocks/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp2n
5blocks/blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp5blocks/blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp2b
/blocks/blocks/b3/blocks/b3/prelu/ReadVariableOp/blocks/blocks/b3/blocks/b3/prelu/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2D
 dense_out/BiasAdd/ReadVariableOp dense_out/BiasAdd/ReadVariableOp2B
dense_out/MatMul/ReadVariableOpdense_out/MatMul/ReadVariableOp2N
%fc_batchnorm/batchnorm/ReadVariableOp%fc_batchnorm/batchnorm/ReadVariableOp2R
'fc_batchnorm/batchnorm/ReadVariableOp_1'fc_batchnorm/batchnorm/ReadVariableOp_12R
'fc_batchnorm/batchnorm/ReadVariableOp_2'fc_batchnorm/batchnorm/ReadVariableOp_22V
)fc_batchnorm/batchnorm/mul/ReadVariableOp)fc_batchnorm/batchnorm/mul/ReadVariableOp22
fc_prelu/ReadVariableOpfc_prelu/ReadVariableOp2P
&featurewise_std/Reshape/ReadVariableOp&featurewise_std/Reshape/ReadVariableOp2T
(featurewise_std/Reshape_1/ReadVariableOp(featurewise_std/Reshape_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
ü£
$
D__inference_simplecnn_layer_call_and_return_conditional_losses_51548

inputsA
/featurewise_std_reshape_readvariableop_resource:``C
1featurewise_std_reshape_1_readvariableop_resource:``X
>blocks_blocks_b1_blocks_b1_conv_conv2d_readvariableop_resource:M
?blocks_blocks_b1_blocks_b1_conv_biasadd_readvariableop_resource:J
<blocks_blocks_b1_blocks_b1_batchnorm_readvariableop_resource:L
>blocks_blocks_b1_blocks_b1_batchnorm_readvariableop_1_resource:[
Mblocks_blocks_b1_blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_resource:]
Oblocks_blocks_b1_blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_1_resource:N
8blocks_blocks_b1_blocks_b1_prelu_readvariableop_resource:`X
>blocks_blocks_b2_blocks_b2_conv_conv2d_readvariableop_resource:M
?blocks_blocks_b2_blocks_b2_conv_biasadd_readvariableop_resource:J
<blocks_blocks_b2_blocks_b2_batchnorm_readvariableop_resource:L
>blocks_blocks_b2_blocks_b2_batchnorm_readvariableop_1_resource:[
Mblocks_blocks_b2_blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_resource:]
Oblocks_blocks_b2_blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_1_resource:N
8blocks_blocks_b2_blocks_b2_prelu_readvariableop_resource:0X
>blocks_blocks_b3_blocks_b3_conv_conv2d_readvariableop_resource:M
?blocks_blocks_b3_blocks_b3_conv_biasadd_readvariableop_resource:J
<blocks_blocks_b3_blocks_b3_batchnorm_readvariableop_resource:L
>blocks_blocks_b3_blocks_b3_batchnorm_readvariableop_1_resource:[
Mblocks_blocks_b3_blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_resource:]
Oblocks_blocks_b3_blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_1_resource:N
8blocks_blocks_b3_blocks_b3_prelu_readvariableop_resource:6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:B
4fc_batchnorm_assignmovingavg_readvariableop_resource:D
6fc_batchnorm_assignmovingavg_1_readvariableop_resource:@
2fc_batchnorm_batchnorm_mul_readvariableop_resource:<
.fc_batchnorm_batchnorm_readvariableop_resource:.
 fc_prelu_readvariableop_resource::
(dense_out_matmul_readvariableop_resource:7
)dense_out_biasadd_readvariableop_resource:
identity¢3blocks/blocks/b1/blocks/b1/batchnorm/AssignNewValue¢5blocks/blocks/b1/blocks/b1/batchnorm/AssignNewValue_1¢Dblocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp¢Fblocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1¢3blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp¢5blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp_1¢6blocks/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp¢5blocks/blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOp¢/blocks/blocks/b1/blocks/b1/prelu/ReadVariableOp¢3blocks/blocks/b2/blocks/b2/batchnorm/AssignNewValue¢5blocks/blocks/b2/blocks/b2/batchnorm/AssignNewValue_1¢Dblocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp¢Fblocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1¢3blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp¢5blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp_1¢6blocks/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp¢5blocks/blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp¢/blocks/blocks/b2/blocks/b2/prelu/ReadVariableOp¢3blocks/blocks/b3/blocks/b3/batchnorm/AssignNewValue¢5blocks/blocks/b3/blocks/b3/batchnorm/AssignNewValue_1¢Dblocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp¢Fblocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1¢3blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp¢5blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp_1¢6blocks/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp¢5blocks/blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp¢/blocks/blocks/b3/blocks/b3/prelu/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢ dense_out/BiasAdd/ReadVariableOp¢dense_out/MatMul/ReadVariableOp¢fc_batchnorm/AssignMovingAvg¢+fc_batchnorm/AssignMovingAvg/ReadVariableOp¢fc_batchnorm/AssignMovingAvg_1¢-fc_batchnorm/AssignMovingAvg_1/ReadVariableOp¢%fc_batchnorm/batchnorm/ReadVariableOp¢)fc_batchnorm/batchnorm/mul/ReadVariableOp¢fc_prelu/ReadVariableOp¢&featurewise_std/Reshape/ReadVariableOp¢(featurewise_std/Reshape_1/ReadVariableOpÀ
&featurewise_std/Reshape/ReadVariableOpReadVariableOp/featurewise_std_reshape_readvariableop_resource*
_output_shapes

:``*
dtype02(
&featurewise_std/Reshape/ReadVariableOp
featurewise_std/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   `   `   2
featurewise_std/Reshape/shapeÂ
featurewise_std/ReshapeReshape.featurewise_std/Reshape/ReadVariableOp:value:0&featurewise_std/Reshape/shape:output:0*
T0*"
_output_shapes
:``2
featurewise_std/ReshapeÆ
(featurewise_std/Reshape_1/ReadVariableOpReadVariableOp1featurewise_std_reshape_1_readvariableop_resource*
_output_shapes

:``*
dtype02*
(featurewise_std/Reshape_1/ReadVariableOp
featurewise_std/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   `   `   2!
featurewise_std/Reshape_1/shapeÊ
featurewise_std/Reshape_1Reshape0featurewise_std/Reshape_1/ReadVariableOp:value:0(featurewise_std/Reshape_1/shape:output:0*
T0*"
_output_shapes
:``2
featurewise_std/Reshape_1
featurewise_std/subSubinputs featurewise_std/Reshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
featurewise_std/sub
featurewise_std/SqrtSqrt"featurewise_std/Reshape_1:output:0*
T0*"
_output_shapes
:``2
featurewise_std/Sqrt{
featurewise_std/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32
featurewise_std/Maximum/y¨
featurewise_std/MaximumMaximumfeaturewise_std/Sqrt:y:0"featurewise_std/Maximum/y:output:0*
T0*"
_output_shapes
:``2
featurewise_std/Maximum©
featurewise_std/truedivRealDivfeaturewise_std/sub:z:0featurewise_std/Maximum:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
featurewise_std/truedivî
#insert_channel_axis/PartitionedCallPartitionedCallfeaturewise_std/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_484102%
#insert_channel_axis/PartitionedCallõ
5blocks/blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOpReadVariableOp>blocks_blocks_b1_blocks_b1_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype027
5blocks/blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOp©
&blocks/blocks/b1/blocks/b1/conv/Conv2DConv2D,insert_channel_axis/PartitionedCall:output:0=blocks/blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*
paddingSAME*
strides
2(
&blocks/blocks/b1/blocks/b1/conv/Conv2Dì
6blocks/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOpReadVariableOp?blocks_blocks_b1_blocks_b1_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6blocks/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp
'blocks/blocks/b1/blocks/b1/conv/BiasAddBiasAdd/blocks/blocks/b1/blocks/b1/conv/Conv2D:output:0>blocks/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2)
'blocks/blocks/b1/blocks/b1/conv/BiasAddã
3blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOpReadVariableOp<blocks_blocks_b1_blocks_b1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype025
3blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOpé
5blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp_1ReadVariableOp>blocks_blocks_b1_blocks_b1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype027
5blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp_1
Dblocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOpReadVariableOpMblocks_blocks_b1_blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dblocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp
Fblocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOblocks_blocks_b1_blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Fblocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1à
5blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3FusedBatchNormV30blocks/blocks/b1/blocks/b1/conv/BiasAdd:output:0;blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp:value:0=blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp_1:value:0Lblocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp:value:0Nblocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ``:::::*
epsilon%o:*
exponential_avg_factor%
×#<27
5blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3û
3blocks/blocks/b1/blocks/b1/batchnorm/AssignNewValueAssignVariableOpMblocks_blocks_b1_blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_resourceBblocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3:batch_mean:0E^blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype025
3blocks/blocks/b1/blocks/b1/batchnorm/AssignNewValue
5blocks/blocks/b1/blocks/b1/batchnorm/AssignNewValue_1AssignVariableOpOblocks_blocks_b1_blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_1_resourceFblocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3:batch_variance:0G^blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype027
5blocks/blocks/b1/blocks/b1/batchnorm/AssignNewValue_1Ë
%blocks/blocks/b1/blocks/b1/prelu/ReluRelu9blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2'
%blocks/blocks/b1/blocks/b1/prelu/Reluß
/blocks/blocks/b1/blocks/b1/prelu/ReadVariableOpReadVariableOp8blocks_blocks_b1_blocks_b1_prelu_readvariableop_resource*"
_output_shapes
:`*
dtype021
/blocks/blocks/b1/blocks/b1/prelu/ReadVariableOp¹
$blocks/blocks/b1/blocks/b1/prelu/NegNeg7blocks/blocks/b1/blocks/b1/prelu/ReadVariableOp:value:0*
T0*"
_output_shapes
:`2&
$blocks/blocks/b1/blocks/b1/prelu/NegÌ
&blocks/blocks/b1/blocks/b1/prelu/Neg_1Neg9blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2(
&blocks/blocks/b1/blocks/b1/prelu/Neg_1À
'blocks/blocks/b1/blocks/b1/prelu/Relu_1Relu*blocks/blocks/b1/blocks/b1/prelu/Neg_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2)
'blocks/blocks/b1/blocks/b1/prelu/Relu_1î
$blocks/blocks/b1/blocks/b1/prelu/mulMul(blocks/blocks/b1/blocks/b1/prelu/Neg:y:05blocks/blocks/b1/blocks/b1/prelu/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2&
$blocks/blocks/b1/blocks/b1/prelu/mulî
$blocks/blocks/b1/blocks/b1/prelu/addAddV23blocks/blocks/b1/blocks/b1/prelu/Relu:activations:0(blocks/blocks/b1/blocks/b1/prelu/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2&
$blocks/blocks/b1/blocks/b1/prelu/addù
*blocks/blocks/b1/blocks/b1/maxpool/MaxPoolMaxPool(blocks/blocks/b1/blocks/b1/prelu/add:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*
ksize
*
paddingSAME*
strides
2,
*blocks/blocks/b1/blocks/b1/maxpool/MaxPoolõ
5blocks/blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOpReadVariableOp>blocks_blocks_b2_blocks_b2_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype027
5blocks/blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp°
&blocks/blocks/b2/blocks/b2/conv/Conv2DConv2D3blocks/blocks/b1/blocks/b1/maxpool/MaxPool:output:0=blocks/blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*
paddingSAME*
strides
2(
&blocks/blocks/b2/blocks/b2/conv/Conv2Dì
6blocks/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOpReadVariableOp?blocks_blocks_b2_blocks_b2_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6blocks/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp
'blocks/blocks/b2/blocks/b2/conv/BiasAddBiasAdd/blocks/blocks/b2/blocks/b2/conv/Conv2D:output:0>blocks/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002)
'blocks/blocks/b2/blocks/b2/conv/BiasAddã
3blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOpReadVariableOp<blocks_blocks_b2_blocks_b2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype025
3blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOpé
5blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp_1ReadVariableOp>blocks_blocks_b2_blocks_b2_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype027
5blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp_1
Dblocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOpReadVariableOpMblocks_blocks_b2_blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dblocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp
Fblocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOblocks_blocks_b2_blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Fblocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1à
5blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3FusedBatchNormV30blocks/blocks/b2/blocks/b2/conv/BiasAdd:output:0;blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp:value:0=blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp_1:value:0Lblocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp:value:0Nblocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00:::::*
epsilon%o:*
exponential_avg_factor%
×#<27
5blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3û
3blocks/blocks/b2/blocks/b2/batchnorm/AssignNewValueAssignVariableOpMblocks_blocks_b2_blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_resourceBblocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3:batch_mean:0E^blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype025
3blocks/blocks/b2/blocks/b2/batchnorm/AssignNewValue
5blocks/blocks/b2/blocks/b2/batchnorm/AssignNewValue_1AssignVariableOpOblocks_blocks_b2_blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_1_resourceFblocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3:batch_variance:0G^blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype027
5blocks/blocks/b2/blocks/b2/batchnorm/AssignNewValue_1Ë
%blocks/blocks/b2/blocks/b2/prelu/ReluRelu9blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002'
%blocks/blocks/b2/blocks/b2/prelu/Reluß
/blocks/blocks/b2/blocks/b2/prelu/ReadVariableOpReadVariableOp8blocks_blocks_b2_blocks_b2_prelu_readvariableop_resource*"
_output_shapes
:0*
dtype021
/blocks/blocks/b2/blocks/b2/prelu/ReadVariableOp¹
$blocks/blocks/b2/blocks/b2/prelu/NegNeg7blocks/blocks/b2/blocks/b2/prelu/ReadVariableOp:value:0*
T0*"
_output_shapes
:02&
$blocks/blocks/b2/blocks/b2/prelu/NegÌ
&blocks/blocks/b2/blocks/b2/prelu/Neg_1Neg9blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002(
&blocks/blocks/b2/blocks/b2/prelu/Neg_1À
'blocks/blocks/b2/blocks/b2/prelu/Relu_1Relu*blocks/blocks/b2/blocks/b2/prelu/Neg_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002)
'blocks/blocks/b2/blocks/b2/prelu/Relu_1î
$blocks/blocks/b2/blocks/b2/prelu/mulMul(blocks/blocks/b2/blocks/b2/prelu/Neg:y:05blocks/blocks/b2/blocks/b2/prelu/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002&
$blocks/blocks/b2/blocks/b2/prelu/mulî
$blocks/blocks/b2/blocks/b2/prelu/addAddV23blocks/blocks/b2/blocks/b2/prelu/Relu:activations:0(blocks/blocks/b2/blocks/b2/prelu/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002&
$blocks/blocks/b2/blocks/b2/prelu/addù
*blocks/blocks/b2/blocks/b2/maxpool/MaxPoolMaxPool(blocks/blocks/b2/blocks/b2/prelu/add:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2,
*blocks/blocks/b2/blocks/b2/maxpool/MaxPoolõ
5blocks/blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOpReadVariableOp>blocks_blocks_b3_blocks_b3_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype027
5blocks/blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp°
&blocks/blocks/b3/blocks/b3/conv/Conv2DConv2D3blocks/blocks/b2/blocks/b2/maxpool/MaxPool:output:0=blocks/blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2(
&blocks/blocks/b3/blocks/b3/conv/Conv2Dì
6blocks/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOpReadVariableOp?blocks_blocks_b3_blocks_b3_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6blocks/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp
'blocks/blocks/b3/blocks/b3/conv/BiasAddBiasAdd/blocks/blocks/b3/blocks/b3/conv/Conv2D:output:0>blocks/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'blocks/blocks/b3/blocks/b3/conv/BiasAddã
3blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOpReadVariableOp<blocks_blocks_b3_blocks_b3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype025
3blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOpé
5blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp_1ReadVariableOp>blocks_blocks_b3_blocks_b3_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype027
5blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp_1
Dblocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOpReadVariableOpMblocks_blocks_b3_blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dblocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp
Fblocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOblocks_blocks_b3_blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Fblocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1à
5blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3FusedBatchNormV30blocks/blocks/b3/blocks/b3/conv/BiasAdd:output:0;blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp:value:0=blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp_1:value:0Lblocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp:value:0Nblocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<27
5blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3û
3blocks/blocks/b3/blocks/b3/batchnorm/AssignNewValueAssignVariableOpMblocks_blocks_b3_blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_resourceBblocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3:batch_mean:0E^blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype025
3blocks/blocks/b3/blocks/b3/batchnorm/AssignNewValue
5blocks/blocks/b3/blocks/b3/batchnorm/AssignNewValue_1AssignVariableOpOblocks_blocks_b3_blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_1_resourceFblocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3:batch_variance:0G^blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype027
5blocks/blocks/b3/blocks/b3/batchnorm/AssignNewValue_1Ë
%blocks/blocks/b3/blocks/b3/prelu/ReluRelu9blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%blocks/blocks/b3/blocks/b3/prelu/Reluß
/blocks/blocks/b3/blocks/b3/prelu/ReadVariableOpReadVariableOp8blocks_blocks_b3_blocks_b3_prelu_readvariableop_resource*"
_output_shapes
:*
dtype021
/blocks/blocks/b3/blocks/b3/prelu/ReadVariableOp¹
$blocks/blocks/b3/blocks/b3/prelu/NegNeg7blocks/blocks/b3/blocks/b3/prelu/ReadVariableOp:value:0*
T0*"
_output_shapes
:2&
$blocks/blocks/b3/blocks/b3/prelu/NegÌ
&blocks/blocks/b3/blocks/b3/prelu/Neg_1Neg9blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&blocks/blocks/b3/blocks/b3/prelu/Neg_1À
'blocks/blocks/b3/blocks/b3/prelu/Relu_1Relu*blocks/blocks/b3/blocks/b3/prelu/Neg_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'blocks/blocks/b3/blocks/b3/prelu/Relu_1î
$blocks/blocks/b3/blocks/b3/prelu/mulMul(blocks/blocks/b3/blocks/b3/prelu/Neg:y:05blocks/blocks/b3/blocks/b3/prelu/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$blocks/blocks/b3/blocks/b3/prelu/mulî
$blocks/blocks/b3/blocks/b3/prelu/addAddV23blocks/blocks/b3/blocks/b3/prelu/Relu:activations:0(blocks/blocks/b3/blocks/b3/prelu/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$blocks/blocks/b3/blocks/b3/prelu/addù
*blocks/blocks/b3/blocks/b3/maxpool/MaxPoolMaxPool(blocks/blocks/b3/blocks/b3/prelu/add:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2,
*blocks/blocks/b3/blocks/b3/maxpool/MaxPool
$globalavgpool/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2&
$globalavgpool/Mean/reduction_indicesÆ
globalavgpool/MeanMean3blocks/blocks/b3/blocks/b3/maxpool/MaxPool:output:0-globalavgpool/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
globalavgpool/Mean
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulglobalavgpool/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAdd¤
+fc_batchnorm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+fc_batchnorm/moments/mean/reduction_indicesÆ
fc_batchnorm/moments/meanMeandense/BiasAdd:output:04fc_batchnorm/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
fc_batchnorm/moments/mean£
!fc_batchnorm/moments/StopGradientStopGradient"fc_batchnorm/moments/mean:output:0*
T0*
_output_shapes

:2#
!fc_batchnorm/moments/StopGradientÛ
&fc_batchnorm/moments/SquaredDifferenceSquaredDifferencedense/BiasAdd:output:0*fc_batchnorm/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&fc_batchnorm/moments/SquaredDifference¬
/fc_batchnorm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 21
/fc_batchnorm/moments/variance/reduction_indicesæ
fc_batchnorm/moments/varianceMean*fc_batchnorm/moments/SquaredDifference:z:08fc_batchnorm/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
fc_batchnorm/moments/variance§
fc_batchnorm/moments/SqueezeSqueeze"fc_batchnorm/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
fc_batchnorm/moments/Squeeze¯
fc_batchnorm/moments/Squeeze_1Squeeze&fc_batchnorm/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2 
fc_batchnorm/moments/Squeeze_1
"fc_batchnorm/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2$
"fc_batchnorm/AssignMovingAvg/decayË
+fc_batchnorm/AssignMovingAvg/ReadVariableOpReadVariableOp4fc_batchnorm_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02-
+fc_batchnorm/AssignMovingAvg/ReadVariableOpÌ
 fc_batchnorm/AssignMovingAvg/subSub3fc_batchnorm/AssignMovingAvg/ReadVariableOp:value:0%fc_batchnorm/moments/Squeeze:output:0*
T0*
_output_shapes
:2"
 fc_batchnorm/AssignMovingAvg/subÃ
 fc_batchnorm/AssignMovingAvg/mulMul$fc_batchnorm/AssignMovingAvg/sub:z:0+fc_batchnorm/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2"
 fc_batchnorm/AssignMovingAvg/mul
fc_batchnorm/AssignMovingAvgAssignSubVariableOp4fc_batchnorm_assignmovingavg_readvariableop_resource$fc_batchnorm/AssignMovingAvg/mul:z:0,^fc_batchnorm/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
fc_batchnorm/AssignMovingAvg
$fc_batchnorm/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2&
$fc_batchnorm/AssignMovingAvg_1/decayÑ
-fc_batchnorm/AssignMovingAvg_1/ReadVariableOpReadVariableOp6fc_batchnorm_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02/
-fc_batchnorm/AssignMovingAvg_1/ReadVariableOpÔ
"fc_batchnorm/AssignMovingAvg_1/subSub5fc_batchnorm/AssignMovingAvg_1/ReadVariableOp:value:0'fc_batchnorm/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2$
"fc_batchnorm/AssignMovingAvg_1/subË
"fc_batchnorm/AssignMovingAvg_1/mulMul&fc_batchnorm/AssignMovingAvg_1/sub:z:0-fc_batchnorm/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2$
"fc_batchnorm/AssignMovingAvg_1/mul
fc_batchnorm/AssignMovingAvg_1AssignSubVariableOp6fc_batchnorm_assignmovingavg_1_readvariableop_resource&fc_batchnorm/AssignMovingAvg_1/mul:z:0.^fc_batchnorm/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02 
fc_batchnorm/AssignMovingAvg_1
fc_batchnorm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
fc_batchnorm/batchnorm/add/y¶
fc_batchnorm/batchnorm/addAddV2'fc_batchnorm/moments/Squeeze_1:output:0%fc_batchnorm/batchnorm/add/y:output:0*
T0*
_output_shapes
:2
fc_batchnorm/batchnorm/add
fc_batchnorm/batchnorm/RsqrtRsqrtfc_batchnorm/batchnorm/add:z:0*
T0*
_output_shapes
:2
fc_batchnorm/batchnorm/RsqrtÅ
)fc_batchnorm/batchnorm/mul/ReadVariableOpReadVariableOp2fc_batchnorm_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02+
)fc_batchnorm/batchnorm/mul/ReadVariableOp¹
fc_batchnorm/batchnorm/mulMul fc_batchnorm/batchnorm/Rsqrt:y:01fc_batchnorm/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
fc_batchnorm/batchnorm/mul­
fc_batchnorm/batchnorm/mul_1Muldense/BiasAdd:output:0fc_batchnorm/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
fc_batchnorm/batchnorm/mul_1¯
fc_batchnorm/batchnorm/mul_2Mul%fc_batchnorm/moments/Squeeze:output:0fc_batchnorm/batchnorm/mul:z:0*
T0*
_output_shapes
:2
fc_batchnorm/batchnorm/mul_2¹
%fc_batchnorm/batchnorm/ReadVariableOpReadVariableOp.fc_batchnorm_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02'
%fc_batchnorm/batchnorm/ReadVariableOpµ
fc_batchnorm/batchnorm/subSub-fc_batchnorm/batchnorm/ReadVariableOp:value:0 fc_batchnorm/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
fc_batchnorm/batchnorm/sub¹
fc_batchnorm/batchnorm/add_1AddV2 fc_batchnorm/batchnorm/mul_1:z:0fc_batchnorm/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
fc_batchnorm/batchnorm/add_1z
fc_prelu/ReluRelu fc_batchnorm/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
fc_prelu/Relu
fc_prelu/ReadVariableOpReadVariableOp fc_prelu_readvariableop_resource*
_output_shapes
:*
dtype02
fc_prelu/ReadVariableOpi
fc_prelu/NegNegfc_prelu/ReadVariableOp:value:0*
T0*
_output_shapes
:2
fc_prelu/Neg{
fc_prelu/Neg_1Neg fc_batchnorm/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
fc_prelu/Neg_1p
fc_prelu/Relu_1Relufc_prelu/Neg_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
fc_prelu/Relu_1
fc_prelu/mulMulfc_prelu/Neg:y:0fc_prelu/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
fc_prelu/mul
fc_prelu/addAddV2fc_prelu/Relu:activations:0fc_prelu/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
fc_prelu/add«
dense_out/MatMul/ReadVariableOpReadVariableOp(dense_out_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
dense_out/MatMul/ReadVariableOp
dense_out/MatMulMatMulfc_prelu/add:z:0'dense_out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_out/MatMulª
 dense_out/BiasAdd/ReadVariableOpReadVariableOp)dense_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_out/BiasAdd/ReadVariableOp©
dense_out/BiasAddBiasAdddense_out/MatMul:product:0(dense_out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_out/BiasAddÕ
probability/PartitionedCallPartitionedCalldense_out/BiasAdd:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_485422
probability/PartitionedCall
IdentityIdentity$probability/PartitionedCall:output:04^blocks/blocks/b1/blocks/b1/batchnorm/AssignNewValue6^blocks/blocks/b1/blocks/b1/batchnorm/AssignNewValue_1E^blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOpG^blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_14^blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp6^blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp_17^blocks/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp6^blocks/blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOp0^blocks/blocks/b1/blocks/b1/prelu/ReadVariableOp4^blocks/blocks/b2/blocks/b2/batchnorm/AssignNewValue6^blocks/blocks/b2/blocks/b2/batchnorm/AssignNewValue_1E^blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOpG^blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_14^blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp6^blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp_17^blocks/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp6^blocks/blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp0^blocks/blocks/b2/blocks/b2/prelu/ReadVariableOp4^blocks/blocks/b3/blocks/b3/batchnorm/AssignNewValue6^blocks/blocks/b3/blocks/b3/batchnorm/AssignNewValue_1E^blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOpG^blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_14^blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp6^blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp_17^blocks/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp6^blocks/blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp0^blocks/blocks/b3/blocks/b3/prelu/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp!^dense_out/BiasAdd/ReadVariableOp ^dense_out/MatMul/ReadVariableOp^fc_batchnorm/AssignMovingAvg,^fc_batchnorm/AssignMovingAvg/ReadVariableOp^fc_batchnorm/AssignMovingAvg_1.^fc_batchnorm/AssignMovingAvg_1/ReadVariableOp&^fc_batchnorm/batchnorm/ReadVariableOp*^fc_batchnorm/batchnorm/mul/ReadVariableOp^fc_prelu/ReadVariableOp'^featurewise_std/Reshape/ReadVariableOp)^featurewise_std/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
3blocks/blocks/b1/blocks/b1/batchnorm/AssignNewValue3blocks/blocks/b1/blocks/b1/batchnorm/AssignNewValue2n
5blocks/blocks/b1/blocks/b1/batchnorm/AssignNewValue_15blocks/blocks/b1/blocks/b1/batchnorm/AssignNewValue_12
Dblocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOpDblocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp2
Fblocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1Fblocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_12j
3blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp3blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp2n
5blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp_15blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp_12p
6blocks/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp6blocks/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp2n
5blocks/blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOp5blocks/blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOp2b
/blocks/blocks/b1/blocks/b1/prelu/ReadVariableOp/blocks/blocks/b1/blocks/b1/prelu/ReadVariableOp2j
3blocks/blocks/b2/blocks/b2/batchnorm/AssignNewValue3blocks/blocks/b2/blocks/b2/batchnorm/AssignNewValue2n
5blocks/blocks/b2/blocks/b2/batchnorm/AssignNewValue_15blocks/blocks/b2/blocks/b2/batchnorm/AssignNewValue_12
Dblocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOpDblocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp2
Fblocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1Fblocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_12j
3blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp3blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp2n
5blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp_15blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp_12p
6blocks/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp6blocks/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp2n
5blocks/blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp5blocks/blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp2b
/blocks/blocks/b2/blocks/b2/prelu/ReadVariableOp/blocks/blocks/b2/blocks/b2/prelu/ReadVariableOp2j
3blocks/blocks/b3/blocks/b3/batchnorm/AssignNewValue3blocks/blocks/b3/blocks/b3/batchnorm/AssignNewValue2n
5blocks/blocks/b3/blocks/b3/batchnorm/AssignNewValue_15blocks/blocks/b3/blocks/b3/batchnorm/AssignNewValue_12
Dblocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOpDblocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp2
Fblocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1Fblocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_12j
3blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp3blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp2n
5blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp_15blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp_12p
6blocks/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp6blocks/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp2n
5blocks/blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp5blocks/blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp2b
/blocks/blocks/b3/blocks/b3/prelu/ReadVariableOp/blocks/blocks/b3/blocks/b3/prelu/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2D
 dense_out/BiasAdd/ReadVariableOp dense_out/BiasAdd/ReadVariableOp2B
dense_out/MatMul/ReadVariableOpdense_out/MatMul/ReadVariableOp2<
fc_batchnorm/AssignMovingAvgfc_batchnorm/AssignMovingAvg2Z
+fc_batchnorm/AssignMovingAvg/ReadVariableOp+fc_batchnorm/AssignMovingAvg/ReadVariableOp2@
fc_batchnorm/AssignMovingAvg_1fc_batchnorm/AssignMovingAvg_12^
-fc_batchnorm/AssignMovingAvg_1/ReadVariableOp-fc_batchnorm/AssignMovingAvg_1/ReadVariableOp2N
%fc_batchnorm/batchnorm/ReadVariableOp%fc_batchnorm/batchnorm/ReadVariableOp2V
)fc_batchnorm/batchnorm/mul/ReadVariableOp)fc_batchnorm/batchnorm/mul/ReadVariableOp22
fc_prelu/ReadVariableOpfc_prelu/ReadVariableOp2P
&featurewise_std/Reshape/ReadVariableOp&featurewise_std/Reshape/ReadVariableOp2T
(featurewise_std/Reshape_1/ReadVariableOp(featurewise_std/Reshape_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
«
h
L__inference_blocks/b2/maxpool_layer_call_and_return_conditional_losses_49106

inputs
identity¬
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


N__inference_blocks/b1/batchnorm_layer_call_and_return_conditional_losses_52362

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é

D__inference_blocks/b1_layer_call_and_return_conditional_losses_48873

inputs.
blocks_b1_conv_48854:"
blocks_b1_conv_48856:'
blocks_b1_batchnorm_48859:'
blocks_b1_batchnorm_48861:'
blocks_b1_batchnorm_48863:'
blocks_b1_batchnorm_48865:+
blocks_b1_prelu_48868:`
identity¢+blocks/b1/batchnorm/StatefulPartitionedCall¢&blocks/b1/conv/StatefulPartitionedCall¢'blocks/b1/prelu/StatefulPartitionedCallº
&blocks/b1/conv/StatefulPartitionedCallStatefulPartitionedCallinputsblocks_b1_conv_48854blocks_b1_conv_48856*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_blocks/b1/conv_layer_call_and_return_conditional_losses_487212(
&blocks/b1/conv/StatefulPartitionedCall´
+blocks/b1/batchnorm/StatefulPartitionedCallStatefulPartitionedCall/blocks/b1/conv/StatefulPartitionedCall:output:0blocks_b1_batchnorm_48859blocks_b1_batchnorm_48861blocks_b1_batchnorm_48863blocks_b1_batchnorm_48865*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_blocks/b1/batchnorm_layer_call_and_return_conditional_losses_488092-
+blocks/b1/batchnorm/StatefulPartitionedCallÔ
'blocks/b1/prelu/StatefulPartitionedCallStatefulPartitionedCall4blocks/b1/batchnorm/StatefulPartitionedCall:output:0blocks_b1_prelu_48868*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_blocks/b1/prelu_layer_call_and_return_conditional_losses_486842)
'blocks/b1/prelu/StatefulPartitionedCall£
!blocks/b1/maxpool/PartitionedCallPartitionedCall0blocks/b1/prelu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_blocks/b1/maxpool_layer_call_and_return_conditional_losses_486982#
!blocks/b1/maxpool/PartitionedCall
IdentityIdentity*blocks/b1/maxpool/PartitionedCall:output:0,^blocks/b1/batchnorm/StatefulPartitionedCall'^blocks/b1/conv/StatefulPartitionedCall(^blocks/b1/prelu/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ``: : : : : : : 2Z
+blocks/b1/batchnorm/StatefulPartitionedCall+blocks/b1/batchnorm/StatefulPartitionedCall2P
&blocks/b1/conv/StatefulPartitionedCall&blocks/b1/conv/StatefulPartitionedCall2R
'blocks/b1/prelu/StatefulPartitionedCall'blocks/b1/prelu/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs

¦
G__inference_fc_batchnorm_layer_call_and_return_conditional_losses_51889

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1Û
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸B

D__inference_simplecnn_layer_call_and_return_conditional_losses_51036
input_spA
/featurewise_std_reshape_readvariableop_resource:``C
1featurewise_std_reshape_1_readvariableop_resource:``&
blocks_50967:
blocks_50969:
blocks_50971:
blocks_50973:
blocks_50975:
blocks_50977:"
blocks_50979:`&
blocks_50981:
blocks_50983:
blocks_50985:
blocks_50987:
blocks_50989:
blocks_50991:"
blocks_50993:0&
blocks_50995:
blocks_50997:
blocks_50999:
blocks_51001:
blocks_51003:
blocks_51005:"
blocks_51007:
dense_51011:
dense_51013: 
fc_batchnorm_51016: 
fc_batchnorm_51018: 
fc_batchnorm_51020: 
fc_batchnorm_51022:
fc_prelu_51026:!
dense_out_51029:
dense_out_51031:
identity¢blocks/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢!dense_out/StatefulPartitionedCall¢$fc_batchnorm/StatefulPartitionedCall¢ fc_prelu/StatefulPartitionedCall¢&featurewise_std/Reshape/ReadVariableOp¢(featurewise_std/Reshape_1/ReadVariableOpÀ
&featurewise_std/Reshape/ReadVariableOpReadVariableOp/featurewise_std_reshape_readvariableop_resource*
_output_shapes

:``*
dtype02(
&featurewise_std/Reshape/ReadVariableOp
featurewise_std/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   `   `   2
featurewise_std/Reshape/shapeÂ
featurewise_std/ReshapeReshape.featurewise_std/Reshape/ReadVariableOp:value:0&featurewise_std/Reshape/shape:output:0*
T0*"
_output_shapes
:``2
featurewise_std/ReshapeÆ
(featurewise_std/Reshape_1/ReadVariableOpReadVariableOp1featurewise_std_reshape_1_readvariableop_resource*
_output_shapes

:``*
dtype02*
(featurewise_std/Reshape_1/ReadVariableOp
featurewise_std/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   `   `   2!
featurewise_std/Reshape_1/shapeÊ
featurewise_std/Reshape_1Reshape0featurewise_std/Reshape_1/ReadVariableOp:value:0(featurewise_std/Reshape_1/shape:output:0*
T0*"
_output_shapes
:``2
featurewise_std/Reshape_1
featurewise_std/subSubinput_sp featurewise_std/Reshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
featurewise_std/sub
featurewise_std/SqrtSqrt"featurewise_std/Reshape_1:output:0*
T0*"
_output_shapes
:``2
featurewise_std/Sqrt{
featurewise_std/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32
featurewise_std/Maximum/y¨
featurewise_std/MaximumMaximumfeaturewise_std/Sqrt:y:0"featurewise_std/Maximum/y:output:0*
T0*"
_output_shapes
:``2
featurewise_std/Maximum©
featurewise_std/truedivRealDivfeaturewise_std/sub:z:0featurewise_std/Maximum:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
featurewise_std/truedivî
#insert_channel_axis/PartitionedCallPartitionedCallfeaturewise_std/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_484102%
#insert_channel_axis/PartitionedCallâ
blocks/StatefulPartitionedCallStatefulPartitionedCall,insert_channel_axis/PartitionedCall:output:0blocks_50967blocks_50969blocks_50971blocks_50973blocks_50975blocks_50977blocks_50979blocks_50981blocks_50983blocks_50985blocks_50987blocks_50989blocks_50991blocks_50993blocks_50995blocks_50997blocks_50999blocks_51001blocks_51003blocks_51005blocks_51007*!
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_blocks_layer_call_and_return_conditional_losses_499652 
blocks/StatefulPartitionedCall
globalavgpool/PartitionedCallPartitionedCall'blocks/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_globalavgpool_layer_call_and_return_conditional_losses_501622
globalavgpool/PartitionedCall¥
dense/StatefulPartitionedCallStatefulPartitionedCall&globalavgpool/PartitionedCall:output:0dense_51011dense_51013*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_504262
dense/StatefulPartitionedCallò
$fc_batchnorm/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0fc_batchnorm_51016fc_batchnorm_51018fc_batchnorm_51020fc_batchnorm_51022*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_fc_batchnorm_layer_call_and_return_conditional_losses_502522&
$fc_batchnorm/StatefulPartitionedCall
fc_dropout/PartitionedCallPartitionedCall-fc_batchnorm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_fc_dropout_layer_call_and_return_conditional_losses_505582
fc_dropout/PartitionedCall
 fc_prelu/StatefulPartitionedCallStatefulPartitionedCall#fc_dropout/PartitionedCall:output:0fc_prelu_51026*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_fc_prelu_layer_call_and_return_conditional_losses_503432"
 fc_prelu/StatefulPartitionedCall¼
!dense_out/StatefulPartitionedCallStatefulPartitionedCall)fc_prelu/StatefulPartitionedCall:output:0dense_out_51029dense_out_51031*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_out_layer_call_and_return_conditional_losses_504612#
!dense_out/StatefulPartitionedCallå
probability/PartitionedCallPartitionedCall*dense_out/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_485422
probability/PartitionedCallû
IdentityIdentity$probability/PartitionedCall:output:0^blocks/StatefulPartitionedCall^dense/StatefulPartitionedCall"^dense_out/StatefulPartitionedCall%^fc_batchnorm/StatefulPartitionedCall!^fc_prelu/StatefulPartitionedCall'^featurewise_std/Reshape/ReadVariableOp)^featurewise_std/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
blocks/StatefulPartitionedCallblocks/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!dense_out/StatefulPartitionedCall!dense_out/StatefulPartitionedCall2L
$fc_batchnorm/StatefulPartitionedCall$fc_batchnorm/StatefulPartitionedCall2D
 fc_prelu/StatefulPartitionedCall fc_prelu/StatefulPartitionedCall2P
&featurewise_std/Reshape/ReadVariableOp&featurewise_std/Reshape/ReadVariableOp2T
(featurewise_std/Reshape_1/ReadVariableOp(featurewise_std/Reshape_1/ReadVariableOp:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
"
_user_specified_name
input_sp
í
O
3__inference_insert_channel_axis_layer_call_fn_42951

inputs
identityÙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_insert_channel_axis_layer_call_and_return_conditional_losses_429462
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ``:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
µ


I__inference_blocks/b3/conv_layer_call_and_return_conditional_losses_52578

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À

N__inference_blocks/b3/batchnorm_layer_call_and_return_conditional_losses_52684

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ú
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
M
1__inference_blocks/b3/maxpool_layer_call_fn_49520

inputs
identityð
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_blocks/b3/maxpool_layer_call_and_return_conditional_losses_495142
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð	
õ
D__inference_dense_out_layer_call_and_return_conditional_losses_51961

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë

D__inference_blocks/b1_layer_call_and_return_conditional_losses_48759

inputs.
blocks_b1_conv_48722:"
blocks_b1_conv_48724:'
blocks_b1_batchnorm_48745:'
blocks_b1_batchnorm_48747:'
blocks_b1_batchnorm_48749:'
blocks_b1_batchnorm_48751:+
blocks_b1_prelu_48754:`
identity¢+blocks/b1/batchnorm/StatefulPartitionedCall¢&blocks/b1/conv/StatefulPartitionedCall¢'blocks/b1/prelu/StatefulPartitionedCallº
&blocks/b1/conv/StatefulPartitionedCallStatefulPartitionedCallinputsblocks_b1_conv_48722blocks_b1_conv_48724*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_blocks/b1/conv_layer_call_and_return_conditional_losses_487212(
&blocks/b1/conv/StatefulPartitionedCall¶
+blocks/b1/batchnorm/StatefulPartitionedCallStatefulPartitionedCall/blocks/b1/conv/StatefulPartitionedCall:output:0blocks_b1_batchnorm_48745blocks_b1_batchnorm_48747blocks_b1_batchnorm_48749blocks_b1_batchnorm_48751*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_blocks/b1/batchnorm_layer_call_and_return_conditional_losses_487442-
+blocks/b1/batchnorm/StatefulPartitionedCallÔ
'blocks/b1/prelu/StatefulPartitionedCallStatefulPartitionedCall4blocks/b1/batchnorm/StatefulPartitionedCall:output:0blocks_b1_prelu_48754*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_blocks/b1/prelu_layer_call_and_return_conditional_losses_486842)
'blocks/b1/prelu/StatefulPartitionedCall£
!blocks/b1/maxpool/PartitionedCallPartitionedCall0blocks/b1/prelu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_blocks/b1/maxpool_layer_call_and_return_conditional_losses_486982#
!blocks/b1/maxpool/PartitionedCall
IdentityIdentity*blocks/b1/maxpool/PartitionedCall:output:0,^blocks/b1/batchnorm/StatefulPartitionedCall'^blocks/b1/conv/StatefulPartitionedCall(^blocks/b1/prelu/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ``: : : : : : : 2Z
+blocks/b1/batchnorm/StatefulPartitionedCall+blocks/b1/batchnorm/StatefulPartitionedCall2P
&blocks/b1/conv/StatefulPartitionedCall&blocks/b1/conv/StatefulPartitionedCall2R
'blocks/b1/prelu/StatefulPartitionedCall'blocks/b1/prelu/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
À

N__inference_blocks/b2/batchnorm_layer_call_and_return_conditional_losses_52541

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ú
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ00: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
 
_user_specified_nameinputs

a
E__inference_fc_dropout_layer_call_and_return_conditional_losses_51942

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ


I__inference_blocks/b3/conv_layer_call_and_return_conditional_losses_49537

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÀB

D__inference_simplecnn_layer_call_and_return_conditional_losses_50950
input_spA
/featurewise_std_reshape_readvariableop_resource:``C
1featurewise_std_reshape_1_readvariableop_resource:``&
blocks_50881:
blocks_50883:
blocks_50885:
blocks_50887:
blocks_50889:
blocks_50891:"
blocks_50893:`&
blocks_50895:
blocks_50897:
blocks_50899:
blocks_50901:
blocks_50903:
blocks_50905:"
blocks_50907:0&
blocks_50909:
blocks_50911:
blocks_50913:
blocks_50915:
blocks_50917:
blocks_50919:"
blocks_50921:
dense_50925:
dense_50927: 
fc_batchnorm_50930: 
fc_batchnorm_50932: 
fc_batchnorm_50934: 
fc_batchnorm_50936:
fc_prelu_50940:!
dense_out_50943:
dense_out_50945:
identity¢blocks/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢!dense_out/StatefulPartitionedCall¢$fc_batchnorm/StatefulPartitionedCall¢ fc_prelu/StatefulPartitionedCall¢&featurewise_std/Reshape/ReadVariableOp¢(featurewise_std/Reshape_1/ReadVariableOpÀ
&featurewise_std/Reshape/ReadVariableOpReadVariableOp/featurewise_std_reshape_readvariableop_resource*
_output_shapes

:``*
dtype02(
&featurewise_std/Reshape/ReadVariableOp
featurewise_std/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   `   `   2
featurewise_std/Reshape/shapeÂ
featurewise_std/ReshapeReshape.featurewise_std/Reshape/ReadVariableOp:value:0&featurewise_std/Reshape/shape:output:0*
T0*"
_output_shapes
:``2
featurewise_std/ReshapeÆ
(featurewise_std/Reshape_1/ReadVariableOpReadVariableOp1featurewise_std_reshape_1_readvariableop_resource*
_output_shapes

:``*
dtype02*
(featurewise_std/Reshape_1/ReadVariableOp
featurewise_std/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   `   `   2!
featurewise_std/Reshape_1/shapeÊ
featurewise_std/Reshape_1Reshape0featurewise_std/Reshape_1/ReadVariableOp:value:0(featurewise_std/Reshape_1/shape:output:0*
T0*"
_output_shapes
:``2
featurewise_std/Reshape_1
featurewise_std/subSubinput_sp featurewise_std/Reshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
featurewise_std/sub
featurewise_std/SqrtSqrt"featurewise_std/Reshape_1:output:0*
T0*"
_output_shapes
:``2
featurewise_std/Sqrt{
featurewise_std/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32
featurewise_std/Maximum/y¨
featurewise_std/MaximumMaximumfeaturewise_std/Sqrt:y:0"featurewise_std/Maximum/y:output:0*
T0*"
_output_shapes
:``2
featurewise_std/Maximum©
featurewise_std/truedivRealDivfeaturewise_std/sub:z:0featurewise_std/Maximum:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
featurewise_std/truedivî
#insert_channel_axis/PartitionedCallPartitionedCallfeaturewise_std/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_484102%
#insert_channel_axis/PartitionedCallè
blocks/StatefulPartitionedCallStatefulPartitionedCall,insert_channel_axis/PartitionedCall:output:0blocks_50881blocks_50883blocks_50885blocks_50887blocks_50889blocks_50891blocks_50893blocks_50895blocks_50897blocks_50899blocks_50901blocks_50903blocks_50905blocks_50907blocks_50909blocks_50911blocks_50913blocks_50915blocks_50917blocks_50919blocks_50921*!
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*7
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_blocks_layer_call_and_return_conditional_losses_498222 
blocks/StatefulPartitionedCall
globalavgpool/PartitionedCallPartitionedCall'blocks/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_globalavgpool_layer_call_and_return_conditional_losses_501622
globalavgpool/PartitionedCall¥
dense/StatefulPartitionedCallStatefulPartitionedCall&globalavgpool/PartitionedCall:output:0dense_50925dense_50927*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_504262
dense/StatefulPartitionedCallô
$fc_batchnorm/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0fc_batchnorm_50930fc_batchnorm_50932fc_batchnorm_50934fc_batchnorm_50936*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_fc_batchnorm_layer_call_and_return_conditional_losses_501922&
$fc_batchnorm/StatefulPartitionedCall
fc_dropout/PartitionedCallPartitionedCall-fc_batchnorm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_fc_dropout_layer_call_and_return_conditional_losses_504462
fc_dropout/PartitionedCall
 fc_prelu/StatefulPartitionedCallStatefulPartitionedCall#fc_dropout/PartitionedCall:output:0fc_prelu_50940*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_fc_prelu_layer_call_and_return_conditional_losses_503432"
 fc_prelu/StatefulPartitionedCall¼
!dense_out/StatefulPartitionedCallStatefulPartitionedCall)fc_prelu/StatefulPartitionedCall:output:0dense_out_50943dense_out_50945*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_out_layer_call_and_return_conditional_losses_504612#
!dense_out/StatefulPartitionedCallå
probability/PartitionedCallPartitionedCall*dense_out/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_485422
probability/PartitionedCallû
IdentityIdentity$probability/PartitionedCall:output:0^blocks/StatefulPartitionedCall^dense/StatefulPartitionedCall"^dense_out/StatefulPartitionedCall%^fc_batchnorm/StatefulPartitionedCall!^fc_prelu/StatefulPartitionedCall'^featurewise_std/Reshape/ReadVariableOp)^featurewise_std/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
blocks/StatefulPartitionedCallblocks/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!dense_out/StatefulPartitionedCall!dense_out/StatefulPartitionedCall2L
$fc_batchnorm/StatefulPartitionedCall$fc_batchnorm/StatefulPartitionedCall2D
 fc_prelu/StatefulPartitionedCall fc_prelu/StatefulPartitionedCall2P
&featurewise_std/Reshape/ReadVariableOp&featurewise_std/Reshape/ReadVariableOp2T
(featurewise_std/Reshape_1/ReadVariableOp(featurewise_std/Reshape_1/ReadVariableOp:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
"
_user_specified_name
input_sp
	
£
)__inference_blocks/b1_layer_call_fn_51980

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:`
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_blocks/b1_layer_call_and_return_conditional_losses_487592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ``: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
ô
½
N__inference_blocks/b3/batchnorm_layer_call_and_return_conditional_losses_49625

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1þ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À	
±
)__inference_blocks/b1_layer_call_fn_48776
blocks_b1_conv_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:`
identity¢StatefulPartitionedCallÎ
StatefulPartitionedCallStatefulPartitionedCallblocks_b1_conv_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_blocks/b1_layer_call_and_return_conditional_losses_487592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ``: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
.
_user_specified_nameblocks/b1/conv_input
	
£
)__inference_blocks/b3_layer_call_fn_52207

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_blocks/b3_layer_call_and_return_conditional_losses_496892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä
Î
3__inference_blocks/b2/batchnorm_layer_call_fn_52461

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_blocks/b2/batchnorm_layer_call_and_return_conditional_losses_490192
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô
½
N__inference_blocks/b1/batchnorm_layer_call_and_return_conditional_losses_52416

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ``:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1þ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ``: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs

x
(__inference_fc_prelu_layer_call_fn_50351

inputs
unknown:
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_fc_prelu_layer_call_and_return_conditional_losses_503432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì	
ñ
@__inference_dense_layer_call_and_return_conditional_losses_51843

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
h
L__inference_blocks/b1/maxpool_layer_call_and_return_conditional_losses_48698

inputs
identity¬
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À

N__inference_blocks/b2/batchnorm_layer_call_and_return_conditional_losses_49152

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ú
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ00: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
 
_user_specified_nameinputs
é

D__inference_blocks/b2_layer_call_and_return_conditional_losses_49281

inputs.
blocks_b2_conv_49262:"
blocks_b2_conv_49264:'
blocks_b2_batchnorm_49267:'
blocks_b2_batchnorm_49269:'
blocks_b2_batchnorm_49271:'
blocks_b2_batchnorm_49273:+
blocks_b2_prelu_49276:0
identity¢+blocks/b2/batchnorm/StatefulPartitionedCall¢&blocks/b2/conv/StatefulPartitionedCall¢'blocks/b2/prelu/StatefulPartitionedCallº
&blocks/b2/conv/StatefulPartitionedCallStatefulPartitionedCallinputsblocks_b2_conv_49262blocks_b2_conv_49264*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_blocks/b2/conv_layer_call_and_return_conditional_losses_491292(
&blocks/b2/conv/StatefulPartitionedCall´
+blocks/b2/batchnorm/StatefulPartitionedCallStatefulPartitionedCall/blocks/b2/conv/StatefulPartitionedCall:output:0blocks_b2_batchnorm_49267blocks_b2_batchnorm_49269blocks_b2_batchnorm_49271blocks_b2_batchnorm_49273*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_blocks/b2/batchnorm_layer_call_and_return_conditional_losses_492172-
+blocks/b2/batchnorm/StatefulPartitionedCallÔ
'blocks/b2/prelu/StatefulPartitionedCallStatefulPartitionedCall4blocks/b2/batchnorm/StatefulPartitionedCall:output:0blocks_b2_prelu_49276*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_blocks/b2/prelu_layer_call_and_return_conditional_losses_490922)
'blocks/b2/prelu/StatefulPartitionedCall£
!blocks/b2/maxpool/PartitionedCallPartitionedCall0blocks/b2/prelu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_blocks/b2/maxpool_layer_call_and_return_conditional_losses_491062#
!blocks/b2/maxpool/PartitionedCall
IdentityIdentity*blocks/b2/maxpool/PartitionedCall:output:0,^blocks/b2/batchnorm/StatefulPartitionedCall'^blocks/b2/conv/StatefulPartitionedCall(^blocks/b2/prelu/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ00: : : : : : : 2Z
+blocks/b2/batchnorm/StatefulPartitionedCall+blocks/b2/batchnorm/StatefulPartitionedCall2P
&blocks/b2/conv/StatefulPartitionedCall&blocks/b2/conv/StatefulPartitionedCall2R
'blocks/b2/prelu/StatefulPartitionedCall'blocks/b2/prelu/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
 
_user_specified_nameinputs
ü
°
&__inference_blocks_layer_call_fn_51595

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:`#
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11: 

unknown_12:0$

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18: 

unknown_19:
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19*!
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*7
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_blocks_layer_call_and_return_conditional_losses_498222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
î
Ç
,__inference_fc_batchnorm_layer_call_fn_51869

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_fc_batchnorm_layer_call_and_return_conditional_losses_502522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾	
±
)__inference_blocks/b2_layer_call_fn_49317
blocks_b2_conv_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:0
identity¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallblocks_b2_conv_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_blocks/b2_layer_call_and_return_conditional_losses_492812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ00: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
.
_user_specified_nameblocks/b2/conv_input


D__inference_blocks/b2_layer_call_and_return_conditional_losses_49361
blocks_b2_conv_input.
blocks_b2_conv_49342:"
blocks_b2_conv_49344:'
blocks_b2_batchnorm_49347:'
blocks_b2_batchnorm_49349:'
blocks_b2_batchnorm_49351:'
blocks_b2_batchnorm_49353:+
blocks_b2_prelu_49356:0
identity¢+blocks/b2/batchnorm/StatefulPartitionedCall¢&blocks/b2/conv/StatefulPartitionedCall¢'blocks/b2/prelu/StatefulPartitionedCallÈ
&blocks/b2/conv/StatefulPartitionedCallStatefulPartitionedCallblocks_b2_conv_inputblocks_b2_conv_49342blocks_b2_conv_49344*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_blocks/b2/conv_layer_call_and_return_conditional_losses_491292(
&blocks/b2/conv/StatefulPartitionedCall´
+blocks/b2/batchnorm/StatefulPartitionedCallStatefulPartitionedCall/blocks/b2/conv/StatefulPartitionedCall:output:0blocks_b2_batchnorm_49347blocks_b2_batchnorm_49349blocks_b2_batchnorm_49351blocks_b2_batchnorm_49353*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_blocks/b2/batchnorm_layer_call_and_return_conditional_losses_492172-
+blocks/b2/batchnorm/StatefulPartitionedCallÔ
'blocks/b2/prelu/StatefulPartitionedCallStatefulPartitionedCall4blocks/b2/batchnorm/StatefulPartitionedCall:output:0blocks_b2_prelu_49356*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_blocks/b2/prelu_layer_call_and_return_conditional_losses_490922)
'blocks/b2/prelu/StatefulPartitionedCall£
!blocks/b2/maxpool/PartitionedCallPartitionedCall0blocks/b2/prelu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_blocks/b2/maxpool_layer_call_and_return_conditional_losses_491062#
!blocks/b2/maxpool/PartitionedCall
IdentityIdentity*blocks/b2/maxpool/PartitionedCall:output:0,^blocks/b2/batchnorm/StatefulPartitionedCall'^blocks/b2/conv/StatefulPartitionedCall(^blocks/b2/prelu/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ00: : : : : : : 2Z
+blocks/b2/batchnorm/StatefulPartitionedCall+blocks/b2/batchnorm/StatefulPartitionedCall2P
&blocks/b2/conv/StatefulPartitionedCall&blocks/b2/conv/StatefulPartitionedCall2R
'blocks/b2/prelu/StatefulPartitionedCall'blocks/b2/prelu/StatefulPartitionedCall:e a
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
.
_user_specified_nameblocks/b2/conv_input
¼

A__inference_blocks_layer_call_and_return_conditional_losses_50155
blocks_input)
blocks_b1_50109:
blocks_b1_50111:
blocks_b1_50113:
blocks_b1_50115:
blocks_b1_50117:
blocks_b1_50119:%
blocks_b1_50121:`)
blocks_b2_50124:
blocks_b2_50126:
blocks_b2_50128:
blocks_b2_50130:
blocks_b2_50132:
blocks_b2_50134:%
blocks_b2_50136:0)
blocks_b3_50139:
blocks_b3_50141:
blocks_b3_50143:
blocks_b3_50145:
blocks_b3_50147:
blocks_b3_50149:%
blocks_b3_50151:
identity¢!blocks/b1/StatefulPartitionedCall¢!blocks/b2/StatefulPartitionedCall¢!blocks/b3/StatefulPartitionedCall
!blocks/b1/StatefulPartitionedCallStatefulPartitionedCallblocks_inputblocks_b1_50109blocks_b1_50111blocks_b1_50113blocks_b1_50115blocks_b1_50117blocks_b1_50119blocks_b1_50121*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_blocks/b1_layer_call_and_return_conditional_losses_488732#
!blocks/b1/StatefulPartitionedCall¢
!blocks/b2/StatefulPartitionedCallStatefulPartitionedCall*blocks/b1/StatefulPartitionedCall:output:0blocks_b2_50124blocks_b2_50126blocks_b2_50128blocks_b2_50130blocks_b2_50132blocks_b2_50134blocks_b2_50136*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_blocks/b2_layer_call_and_return_conditional_losses_492812#
!blocks/b2/StatefulPartitionedCall¢
!blocks/b3/StatefulPartitionedCallStatefulPartitionedCall*blocks/b2/StatefulPartitionedCall:output:0blocks_b3_50139blocks_b3_50141blocks_b3_50143blocks_b3_50145blocks_b3_50147blocks_b3_50149blocks_b3_50151*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_blocks/b3_layer_call_and_return_conditional_losses_496892#
!blocks/b3/StatefulPartitionedCallò
IdentityIdentity*blocks/b3/StatefulPartitionedCall:output:0"^blocks/b1/StatefulPartitionedCall"^blocks/b2/StatefulPartitionedCall"^blocks/b3/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : 2F
!blocks/b1/StatefulPartitionedCall!blocks/b1/StatefulPartitionedCall2F
!blocks/b2/StatefulPartitionedCall!blocks/b2/StatefulPartitionedCall2F
!blocks/b3/StatefulPartitionedCall!blocks/b3/StatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
&
_user_specified_nameblocks/input
©
Ã%
 __inference__wrapped_model_48545
input_spK
9simplecnn_featurewise_std_reshape_readvariableop_resource:``M
;simplecnn_featurewise_std_reshape_1_readvariableop_resource:``b
Hsimplecnn_blocks_blocks_b1_blocks_b1_conv_conv2d_readvariableop_resource:W
Isimplecnn_blocks_blocks_b1_blocks_b1_conv_biasadd_readvariableop_resource:T
Fsimplecnn_blocks_blocks_b1_blocks_b1_batchnorm_readvariableop_resource:V
Hsimplecnn_blocks_blocks_b1_blocks_b1_batchnorm_readvariableop_1_resource:e
Wsimplecnn_blocks_blocks_b1_blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_resource:g
Ysimplecnn_blocks_blocks_b1_blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_1_resource:X
Bsimplecnn_blocks_blocks_b1_blocks_b1_prelu_readvariableop_resource:`b
Hsimplecnn_blocks_blocks_b2_blocks_b2_conv_conv2d_readvariableop_resource:W
Isimplecnn_blocks_blocks_b2_blocks_b2_conv_biasadd_readvariableop_resource:T
Fsimplecnn_blocks_blocks_b2_blocks_b2_batchnorm_readvariableop_resource:V
Hsimplecnn_blocks_blocks_b2_blocks_b2_batchnorm_readvariableop_1_resource:e
Wsimplecnn_blocks_blocks_b2_blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_resource:g
Ysimplecnn_blocks_blocks_b2_blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_1_resource:X
Bsimplecnn_blocks_blocks_b2_blocks_b2_prelu_readvariableop_resource:0b
Hsimplecnn_blocks_blocks_b3_blocks_b3_conv_conv2d_readvariableop_resource:W
Isimplecnn_blocks_blocks_b3_blocks_b3_conv_biasadd_readvariableop_resource:T
Fsimplecnn_blocks_blocks_b3_blocks_b3_batchnorm_readvariableop_resource:V
Hsimplecnn_blocks_blocks_b3_blocks_b3_batchnorm_readvariableop_1_resource:e
Wsimplecnn_blocks_blocks_b3_blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_resource:g
Ysimplecnn_blocks_blocks_b3_blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_1_resource:X
Bsimplecnn_blocks_blocks_b3_blocks_b3_prelu_readvariableop_resource:@
.simplecnn_dense_matmul_readvariableop_resource:=
/simplecnn_dense_biasadd_readvariableop_resource:F
8simplecnn_fc_batchnorm_batchnorm_readvariableop_resource:J
<simplecnn_fc_batchnorm_batchnorm_mul_readvariableop_resource:H
:simplecnn_fc_batchnorm_batchnorm_readvariableop_1_resource:H
:simplecnn_fc_batchnorm_batchnorm_readvariableop_2_resource:8
*simplecnn_fc_prelu_readvariableop_resource:D
2simplecnn_dense_out_matmul_readvariableop_resource:A
3simplecnn_dense_out_biasadd_readvariableop_resource:
identity¢Nsimplecnn/blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp¢Psimplecnn/blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1¢=simplecnn/blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp¢?simplecnn/blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp_1¢@simplecnn/blocks/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp¢?simplecnn/blocks/blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOp¢9simplecnn/blocks/blocks/b1/blocks/b1/prelu/ReadVariableOp¢Nsimplecnn/blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp¢Psimplecnn/blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1¢=simplecnn/blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp¢?simplecnn/blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp_1¢@simplecnn/blocks/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp¢?simplecnn/blocks/blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp¢9simplecnn/blocks/blocks/b2/blocks/b2/prelu/ReadVariableOp¢Nsimplecnn/blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp¢Psimplecnn/blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1¢=simplecnn/blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp¢?simplecnn/blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp_1¢@simplecnn/blocks/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp¢?simplecnn/blocks/blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp¢9simplecnn/blocks/blocks/b3/blocks/b3/prelu/ReadVariableOp¢&simplecnn/dense/BiasAdd/ReadVariableOp¢%simplecnn/dense/MatMul/ReadVariableOp¢*simplecnn/dense_out/BiasAdd/ReadVariableOp¢)simplecnn/dense_out/MatMul/ReadVariableOp¢/simplecnn/fc_batchnorm/batchnorm/ReadVariableOp¢1simplecnn/fc_batchnorm/batchnorm/ReadVariableOp_1¢1simplecnn/fc_batchnorm/batchnorm/ReadVariableOp_2¢3simplecnn/fc_batchnorm/batchnorm/mul/ReadVariableOp¢!simplecnn/fc_prelu/ReadVariableOp¢0simplecnn/featurewise_std/Reshape/ReadVariableOp¢2simplecnn/featurewise_std/Reshape_1/ReadVariableOpÞ
0simplecnn/featurewise_std/Reshape/ReadVariableOpReadVariableOp9simplecnn_featurewise_std_reshape_readvariableop_resource*
_output_shapes

:``*
dtype022
0simplecnn/featurewise_std/Reshape/ReadVariableOp§
'simplecnn/featurewise_std/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   `   `   2)
'simplecnn/featurewise_std/Reshape/shapeê
!simplecnn/featurewise_std/ReshapeReshape8simplecnn/featurewise_std/Reshape/ReadVariableOp:value:00simplecnn/featurewise_std/Reshape/shape:output:0*
T0*"
_output_shapes
:``2#
!simplecnn/featurewise_std/Reshapeä
2simplecnn/featurewise_std/Reshape_1/ReadVariableOpReadVariableOp;simplecnn_featurewise_std_reshape_1_readvariableop_resource*
_output_shapes

:``*
dtype024
2simplecnn/featurewise_std/Reshape_1/ReadVariableOp«
)simplecnn/featurewise_std/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   `   `   2+
)simplecnn/featurewise_std/Reshape_1/shapeò
#simplecnn/featurewise_std/Reshape_1Reshape:simplecnn/featurewise_std/Reshape_1/ReadVariableOp:value:02simplecnn/featurewise_std/Reshape_1/shape:output:0*
T0*"
_output_shapes
:``2%
#simplecnn/featurewise_std/Reshape_1±
simplecnn/featurewise_std/subSubinput_sp*simplecnn/featurewise_std/Reshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
simplecnn/featurewise_std/sub£
simplecnn/featurewise_std/SqrtSqrt,simplecnn/featurewise_std/Reshape_1:output:0*
T0*"
_output_shapes
:``2 
simplecnn/featurewise_std/Sqrt
#simplecnn/featurewise_std/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32%
#simplecnn/featurewise_std/Maximum/yÐ
!simplecnn/featurewise_std/MaximumMaximum"simplecnn/featurewise_std/Sqrt:y:0,simplecnn/featurewise_std/Maximum/y:output:0*
T0*"
_output_shapes
:``2#
!simplecnn/featurewise_std/MaximumÑ
!simplecnn/featurewise_std/truedivRealDiv!simplecnn/featurewise_std/sub:z:0%simplecnn/featurewise_std/Maximum:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2#
!simplecnn/featurewise_std/truediv
-simplecnn/insert_channel_axis/PartitionedCallPartitionedCall%simplecnn/featurewise_std/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_484102/
-simplecnn/insert_channel_axis/PartitionedCall
?simplecnn/blocks/blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOpReadVariableOpHsimplecnn_blocks_blocks_b1_blocks_b1_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02A
?simplecnn/blocks/blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOpÑ
0simplecnn/blocks/blocks/b1/blocks/b1/conv/Conv2DConv2D6simplecnn/insert_channel_axis/PartitionedCall:output:0Gsimplecnn/blocks/blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*
paddingSAME*
strides
22
0simplecnn/blocks/blocks/b1/blocks/b1/conv/Conv2D
@simplecnn/blocks/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOpReadVariableOpIsimplecnn_blocks_blocks_b1_blocks_b1_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02B
@simplecnn/blocks/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp°
1simplecnn/blocks/blocks/b1/blocks/b1/conv/BiasAddBiasAdd9simplecnn/blocks/blocks/b1/blocks/b1/conv/Conv2D:output:0Hsimplecnn/blocks/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``23
1simplecnn/blocks/blocks/b1/blocks/b1/conv/BiasAdd
=simplecnn/blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOpReadVariableOpFsimplecnn_blocks_blocks_b1_blocks_b1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02?
=simplecnn/blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp
?simplecnn/blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp_1ReadVariableOpHsimplecnn_blocks_blocks_b1_blocks_b1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?simplecnn/blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp_1´
Nsimplecnn/blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOpReadVariableOpWsimplecnn_blocks_blocks_b1_blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02P
Nsimplecnn/blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOpº
Psimplecnn/blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYsimplecnn_blocks_blocks_b1_blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02R
Psimplecnn/blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1
?simplecnn/blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3FusedBatchNormV3:simplecnn/blocks/blocks/b1/blocks/b1/conv/BiasAdd:output:0Esimplecnn/blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp:value:0Gsimplecnn/blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp_1:value:0Vsimplecnn/blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp:value:0Xsimplecnn/blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ``:::::*
epsilon%o:*
is_training( 2A
?simplecnn/blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3é
/simplecnn/blocks/blocks/b1/blocks/b1/prelu/ReluReluCsimplecnn/blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``21
/simplecnn/blocks/blocks/b1/blocks/b1/prelu/Reluý
9simplecnn/blocks/blocks/b1/blocks/b1/prelu/ReadVariableOpReadVariableOpBsimplecnn_blocks_blocks_b1_blocks_b1_prelu_readvariableop_resource*"
_output_shapes
:`*
dtype02;
9simplecnn/blocks/blocks/b1/blocks/b1/prelu/ReadVariableOp×
.simplecnn/blocks/blocks/b1/blocks/b1/prelu/NegNegAsimplecnn/blocks/blocks/b1/blocks/b1/prelu/ReadVariableOp:value:0*
T0*"
_output_shapes
:`20
.simplecnn/blocks/blocks/b1/blocks/b1/prelu/Negê
0simplecnn/blocks/blocks/b1/blocks/b1/prelu/Neg_1NegCsimplecnn/blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``22
0simplecnn/blocks/blocks/b1/blocks/b1/prelu/Neg_1Þ
1simplecnn/blocks/blocks/b1/blocks/b1/prelu/Relu_1Relu4simplecnn/blocks/blocks/b1/blocks/b1/prelu/Neg_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``23
1simplecnn/blocks/blocks/b1/blocks/b1/prelu/Relu_1
.simplecnn/blocks/blocks/b1/blocks/b1/prelu/mulMul2simplecnn/blocks/blocks/b1/blocks/b1/prelu/Neg:y:0?simplecnn/blocks/blocks/b1/blocks/b1/prelu/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``20
.simplecnn/blocks/blocks/b1/blocks/b1/prelu/mul
.simplecnn/blocks/blocks/b1/blocks/b1/prelu/addAddV2=simplecnn/blocks/blocks/b1/blocks/b1/prelu/Relu:activations:02simplecnn/blocks/blocks/b1/blocks/b1/prelu/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``20
.simplecnn/blocks/blocks/b1/blocks/b1/prelu/add
4simplecnn/blocks/blocks/b1/blocks/b1/maxpool/MaxPoolMaxPool2simplecnn/blocks/blocks/b1/blocks/b1/prelu/add:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*
ksize
*
paddingSAME*
strides
26
4simplecnn/blocks/blocks/b1/blocks/b1/maxpool/MaxPool
?simplecnn/blocks/blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOpReadVariableOpHsimplecnn_blocks_blocks_b2_blocks_b2_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02A
?simplecnn/blocks/blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOpØ
0simplecnn/blocks/blocks/b2/blocks/b2/conv/Conv2DConv2D=simplecnn/blocks/blocks/b1/blocks/b1/maxpool/MaxPool:output:0Gsimplecnn/blocks/blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*
paddingSAME*
strides
22
0simplecnn/blocks/blocks/b2/blocks/b2/conv/Conv2D
@simplecnn/blocks/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOpReadVariableOpIsimplecnn_blocks_blocks_b2_blocks_b2_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02B
@simplecnn/blocks/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp°
1simplecnn/blocks/blocks/b2/blocks/b2/conv/BiasAddBiasAdd9simplecnn/blocks/blocks/b2/blocks/b2/conv/Conv2D:output:0Hsimplecnn/blocks/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0023
1simplecnn/blocks/blocks/b2/blocks/b2/conv/BiasAdd
=simplecnn/blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOpReadVariableOpFsimplecnn_blocks_blocks_b2_blocks_b2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02?
=simplecnn/blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp
?simplecnn/blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp_1ReadVariableOpHsimplecnn_blocks_blocks_b2_blocks_b2_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?simplecnn/blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp_1´
Nsimplecnn/blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOpReadVariableOpWsimplecnn_blocks_blocks_b2_blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02P
Nsimplecnn/blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOpº
Psimplecnn/blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYsimplecnn_blocks_blocks_b2_blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02R
Psimplecnn/blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1
?simplecnn/blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3FusedBatchNormV3:simplecnn/blocks/blocks/b2/blocks/b2/conv/BiasAdd:output:0Esimplecnn/blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp:value:0Gsimplecnn/blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp_1:value:0Vsimplecnn/blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp:value:0Xsimplecnn/blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00:::::*
epsilon%o:*
is_training( 2A
?simplecnn/blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3é
/simplecnn/blocks/blocks/b2/blocks/b2/prelu/ReluReluCsimplecnn/blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0021
/simplecnn/blocks/blocks/b2/blocks/b2/prelu/Reluý
9simplecnn/blocks/blocks/b2/blocks/b2/prelu/ReadVariableOpReadVariableOpBsimplecnn_blocks_blocks_b2_blocks_b2_prelu_readvariableop_resource*"
_output_shapes
:0*
dtype02;
9simplecnn/blocks/blocks/b2/blocks/b2/prelu/ReadVariableOp×
.simplecnn/blocks/blocks/b2/blocks/b2/prelu/NegNegAsimplecnn/blocks/blocks/b2/blocks/b2/prelu/ReadVariableOp:value:0*
T0*"
_output_shapes
:020
.simplecnn/blocks/blocks/b2/blocks/b2/prelu/Negê
0simplecnn/blocks/blocks/b2/blocks/b2/prelu/Neg_1NegCsimplecnn/blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0022
0simplecnn/blocks/blocks/b2/blocks/b2/prelu/Neg_1Þ
1simplecnn/blocks/blocks/b2/blocks/b2/prelu/Relu_1Relu4simplecnn/blocks/blocks/b2/blocks/b2/prelu/Neg_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0023
1simplecnn/blocks/blocks/b2/blocks/b2/prelu/Relu_1
.simplecnn/blocks/blocks/b2/blocks/b2/prelu/mulMul2simplecnn/blocks/blocks/b2/blocks/b2/prelu/Neg:y:0?simplecnn/blocks/blocks/b2/blocks/b2/prelu/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0020
.simplecnn/blocks/blocks/b2/blocks/b2/prelu/mul
.simplecnn/blocks/blocks/b2/blocks/b2/prelu/addAddV2=simplecnn/blocks/blocks/b2/blocks/b2/prelu/Relu:activations:02simplecnn/blocks/blocks/b2/blocks/b2/prelu/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0020
.simplecnn/blocks/blocks/b2/blocks/b2/prelu/add
4simplecnn/blocks/blocks/b2/blocks/b2/maxpool/MaxPoolMaxPool2simplecnn/blocks/blocks/b2/blocks/b2/prelu/add:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
26
4simplecnn/blocks/blocks/b2/blocks/b2/maxpool/MaxPool
?simplecnn/blocks/blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOpReadVariableOpHsimplecnn_blocks_blocks_b3_blocks_b3_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02A
?simplecnn/blocks/blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOpØ
0simplecnn/blocks/blocks/b3/blocks/b3/conv/Conv2DConv2D=simplecnn/blocks/blocks/b2/blocks/b2/maxpool/MaxPool:output:0Gsimplecnn/blocks/blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
22
0simplecnn/blocks/blocks/b3/blocks/b3/conv/Conv2D
@simplecnn/blocks/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOpReadVariableOpIsimplecnn_blocks_blocks_b3_blocks_b3_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02B
@simplecnn/blocks/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp°
1simplecnn/blocks/blocks/b3/blocks/b3/conv/BiasAddBiasAdd9simplecnn/blocks/blocks/b3/blocks/b3/conv/Conv2D:output:0Hsimplecnn/blocks/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1simplecnn/blocks/blocks/b3/blocks/b3/conv/BiasAdd
=simplecnn/blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOpReadVariableOpFsimplecnn_blocks_blocks_b3_blocks_b3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02?
=simplecnn/blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp
?simplecnn/blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp_1ReadVariableOpHsimplecnn_blocks_blocks_b3_blocks_b3_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?simplecnn/blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp_1´
Nsimplecnn/blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOpReadVariableOpWsimplecnn_blocks_blocks_b3_blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02P
Nsimplecnn/blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOpº
Psimplecnn/blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYsimplecnn_blocks_blocks_b3_blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02R
Psimplecnn/blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1
?simplecnn/blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3FusedBatchNormV3:simplecnn/blocks/blocks/b3/blocks/b3/conv/BiasAdd:output:0Esimplecnn/blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp:value:0Gsimplecnn/blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp_1:value:0Vsimplecnn/blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp:value:0Xsimplecnn/blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2A
?simplecnn/blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3é
/simplecnn/blocks/blocks/b3/blocks/b3/prelu/ReluReluCsimplecnn/blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/simplecnn/blocks/blocks/b3/blocks/b3/prelu/Reluý
9simplecnn/blocks/blocks/b3/blocks/b3/prelu/ReadVariableOpReadVariableOpBsimplecnn_blocks_blocks_b3_blocks_b3_prelu_readvariableop_resource*"
_output_shapes
:*
dtype02;
9simplecnn/blocks/blocks/b3/blocks/b3/prelu/ReadVariableOp×
.simplecnn/blocks/blocks/b3/blocks/b3/prelu/NegNegAsimplecnn/blocks/blocks/b3/blocks/b3/prelu/ReadVariableOp:value:0*
T0*"
_output_shapes
:20
.simplecnn/blocks/blocks/b3/blocks/b3/prelu/Negê
0simplecnn/blocks/blocks/b3/blocks/b3/prelu/Neg_1NegCsimplecnn/blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0simplecnn/blocks/blocks/b3/blocks/b3/prelu/Neg_1Þ
1simplecnn/blocks/blocks/b3/blocks/b3/prelu/Relu_1Relu4simplecnn/blocks/blocks/b3/blocks/b3/prelu/Neg_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1simplecnn/blocks/blocks/b3/blocks/b3/prelu/Relu_1
.simplecnn/blocks/blocks/b3/blocks/b3/prelu/mulMul2simplecnn/blocks/blocks/b3/blocks/b3/prelu/Neg:y:0?simplecnn/blocks/blocks/b3/blocks/b3/prelu/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.simplecnn/blocks/blocks/b3/blocks/b3/prelu/mul
.simplecnn/blocks/blocks/b3/blocks/b3/prelu/addAddV2=simplecnn/blocks/blocks/b3/blocks/b3/prelu/Relu:activations:02simplecnn/blocks/blocks/b3/blocks/b3/prelu/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.simplecnn/blocks/blocks/b3/blocks/b3/prelu/add
4simplecnn/blocks/blocks/b3/blocks/b3/maxpool/MaxPoolMaxPool2simplecnn/blocks/blocks/b3/blocks/b3/prelu/add:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
26
4simplecnn/blocks/blocks/b3/blocks/b3/maxpool/MaxPool±
.simplecnn/globalavgpool/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      20
.simplecnn/globalavgpool/Mean/reduction_indicesî
simplecnn/globalavgpool/MeanMean=simplecnn/blocks/blocks/b3/blocks/b3/maxpool/MaxPool:output:07simplecnn/globalavgpool/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
simplecnn/globalavgpool/Mean½
%simplecnn/dense/MatMul/ReadVariableOpReadVariableOp.simplecnn_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02'
%simplecnn/dense/MatMul/ReadVariableOpÂ
simplecnn/dense/MatMulMatMul%simplecnn/globalavgpool/Mean:output:0-simplecnn/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
simplecnn/dense/MatMul¼
&simplecnn/dense/BiasAdd/ReadVariableOpReadVariableOp/simplecnn_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&simplecnn/dense/BiasAdd/ReadVariableOpÁ
simplecnn/dense/BiasAddBiasAdd simplecnn/dense/MatMul:product:0.simplecnn/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
simplecnn/dense/BiasAdd×
/simplecnn/fc_batchnorm/batchnorm/ReadVariableOpReadVariableOp8simplecnn_fc_batchnorm_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/simplecnn/fc_batchnorm/batchnorm/ReadVariableOp
&simplecnn/fc_batchnorm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&simplecnn/fc_batchnorm/batchnorm/add/yä
$simplecnn/fc_batchnorm/batchnorm/addAddV27simplecnn/fc_batchnorm/batchnorm/ReadVariableOp:value:0/simplecnn/fc_batchnorm/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$simplecnn/fc_batchnorm/batchnorm/add¨
&simplecnn/fc_batchnorm/batchnorm/RsqrtRsqrt(simplecnn/fc_batchnorm/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&simplecnn/fc_batchnorm/batchnorm/Rsqrtã
3simplecnn/fc_batchnorm/batchnorm/mul/ReadVariableOpReadVariableOp<simplecnn_fc_batchnorm_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3simplecnn/fc_batchnorm/batchnorm/mul/ReadVariableOpá
$simplecnn/fc_batchnorm/batchnorm/mulMul*simplecnn/fc_batchnorm/batchnorm/Rsqrt:y:0;simplecnn/fc_batchnorm/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$simplecnn/fc_batchnorm/batchnorm/mulÕ
&simplecnn/fc_batchnorm/batchnorm/mul_1Mul simplecnn/dense/BiasAdd:output:0(simplecnn/fc_batchnorm/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&simplecnn/fc_batchnorm/batchnorm/mul_1Ý
1simplecnn/fc_batchnorm/batchnorm/ReadVariableOp_1ReadVariableOp:simplecnn_fc_batchnorm_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1simplecnn/fc_batchnorm/batchnorm/ReadVariableOp_1á
&simplecnn/fc_batchnorm/batchnorm/mul_2Mul9simplecnn/fc_batchnorm/batchnorm/ReadVariableOp_1:value:0(simplecnn/fc_batchnorm/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&simplecnn/fc_batchnorm/batchnorm/mul_2Ý
1simplecnn/fc_batchnorm/batchnorm/ReadVariableOp_2ReadVariableOp:simplecnn_fc_batchnorm_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1simplecnn/fc_batchnorm/batchnorm/ReadVariableOp_2ß
$simplecnn/fc_batchnorm/batchnorm/subSub9simplecnn/fc_batchnorm/batchnorm/ReadVariableOp_2:value:0*simplecnn/fc_batchnorm/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$simplecnn/fc_batchnorm/batchnorm/subá
&simplecnn/fc_batchnorm/batchnorm/add_1AddV2*simplecnn/fc_batchnorm/batchnorm/mul_1:z:0(simplecnn/fc_batchnorm/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&simplecnn/fc_batchnorm/batchnorm/add_1¨
simplecnn/fc_dropout/IdentityIdentity*simplecnn/fc_batchnorm/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
simplecnn/fc_dropout/Identity
simplecnn/fc_prelu/ReluRelu&simplecnn/fc_dropout/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
simplecnn/fc_prelu/Relu­
!simplecnn/fc_prelu/ReadVariableOpReadVariableOp*simplecnn_fc_prelu_readvariableop_resource*
_output_shapes
:*
dtype02#
!simplecnn/fc_prelu/ReadVariableOp
simplecnn/fc_prelu/NegNeg)simplecnn/fc_prelu/ReadVariableOp:value:0*
T0*
_output_shapes
:2
simplecnn/fc_prelu/Neg
simplecnn/fc_prelu/Neg_1Neg&simplecnn/fc_dropout/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
simplecnn/fc_prelu/Neg_1
simplecnn/fc_prelu/Relu_1Relusimplecnn/fc_prelu/Neg_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
simplecnn/fc_prelu/Relu_1®
simplecnn/fc_prelu/mulMulsimplecnn/fc_prelu/Neg:y:0'simplecnn/fc_prelu/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
simplecnn/fc_prelu/mul®
simplecnn/fc_prelu/addAddV2%simplecnn/fc_prelu/Relu:activations:0simplecnn/fc_prelu/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
simplecnn/fc_prelu/addÉ
)simplecnn/dense_out/MatMul/ReadVariableOpReadVariableOp2simplecnn_dense_out_matmul_readvariableop_resource*
_output_shapes

:*
dtype02+
)simplecnn/dense_out/MatMul/ReadVariableOpÃ
simplecnn/dense_out/MatMulMatMulsimplecnn/fc_prelu/add:z:01simplecnn/dense_out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
simplecnn/dense_out/MatMulÈ
*simplecnn/dense_out/BiasAdd/ReadVariableOpReadVariableOp3simplecnn_dense_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*simplecnn/dense_out/BiasAdd/ReadVariableOpÑ
simplecnn/dense_out/BiasAddBiasAdd$simplecnn/dense_out/MatMul:product:02simplecnn/dense_out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
simplecnn/dense_out/BiasAddó
%simplecnn/probability/PartitionedCallPartitionedCall$simplecnn/dense_out/BiasAdd:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_485422'
%simplecnn/probability/PartitionedCall½
IdentityIdentity.simplecnn/probability/PartitionedCall:output:0O^simplecnn/blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOpQ^simplecnn/blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1>^simplecnn/blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp@^simplecnn/blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp_1A^simplecnn/blocks/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp@^simplecnn/blocks/blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOp:^simplecnn/blocks/blocks/b1/blocks/b1/prelu/ReadVariableOpO^simplecnn/blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOpQ^simplecnn/blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1>^simplecnn/blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp@^simplecnn/blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp_1A^simplecnn/blocks/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp@^simplecnn/blocks/blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp:^simplecnn/blocks/blocks/b2/blocks/b2/prelu/ReadVariableOpO^simplecnn/blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOpQ^simplecnn/blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1>^simplecnn/blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp@^simplecnn/blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp_1A^simplecnn/blocks/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp@^simplecnn/blocks/blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp:^simplecnn/blocks/blocks/b3/blocks/b3/prelu/ReadVariableOp'^simplecnn/dense/BiasAdd/ReadVariableOp&^simplecnn/dense/MatMul/ReadVariableOp+^simplecnn/dense_out/BiasAdd/ReadVariableOp*^simplecnn/dense_out/MatMul/ReadVariableOp0^simplecnn/fc_batchnorm/batchnorm/ReadVariableOp2^simplecnn/fc_batchnorm/batchnorm/ReadVariableOp_12^simplecnn/fc_batchnorm/batchnorm/ReadVariableOp_24^simplecnn/fc_batchnorm/batchnorm/mul/ReadVariableOp"^simplecnn/fc_prelu/ReadVariableOp1^simplecnn/featurewise_std/Reshape/ReadVariableOp3^simplecnn/featurewise_std/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2 
Nsimplecnn/blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOpNsimplecnn/blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp2¤
Psimplecnn/blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1Psimplecnn/blocks/blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_12~
=simplecnn/blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp=simplecnn/blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp2
?simplecnn/blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp_1?simplecnn/blocks/blocks/b1/blocks/b1/batchnorm/ReadVariableOp_12
@simplecnn/blocks/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp@simplecnn/blocks/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp2
?simplecnn/blocks/blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOp?simplecnn/blocks/blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOp2v
9simplecnn/blocks/blocks/b1/blocks/b1/prelu/ReadVariableOp9simplecnn/blocks/blocks/b1/blocks/b1/prelu/ReadVariableOp2 
Nsimplecnn/blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOpNsimplecnn/blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp2¤
Psimplecnn/blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1Psimplecnn/blocks/blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_12~
=simplecnn/blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp=simplecnn/blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp2
?simplecnn/blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp_1?simplecnn/blocks/blocks/b2/blocks/b2/batchnorm/ReadVariableOp_12
@simplecnn/blocks/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp@simplecnn/blocks/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp2
?simplecnn/blocks/blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp?simplecnn/blocks/blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp2v
9simplecnn/blocks/blocks/b2/blocks/b2/prelu/ReadVariableOp9simplecnn/blocks/blocks/b2/blocks/b2/prelu/ReadVariableOp2 
Nsimplecnn/blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOpNsimplecnn/blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp2¤
Psimplecnn/blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1Psimplecnn/blocks/blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_12~
=simplecnn/blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp=simplecnn/blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp2
?simplecnn/blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp_1?simplecnn/blocks/blocks/b3/blocks/b3/batchnorm/ReadVariableOp_12
@simplecnn/blocks/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp@simplecnn/blocks/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp2
?simplecnn/blocks/blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp?simplecnn/blocks/blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp2v
9simplecnn/blocks/blocks/b3/blocks/b3/prelu/ReadVariableOp9simplecnn/blocks/blocks/b3/blocks/b3/prelu/ReadVariableOp2P
&simplecnn/dense/BiasAdd/ReadVariableOp&simplecnn/dense/BiasAdd/ReadVariableOp2N
%simplecnn/dense/MatMul/ReadVariableOp%simplecnn/dense/MatMul/ReadVariableOp2X
*simplecnn/dense_out/BiasAdd/ReadVariableOp*simplecnn/dense_out/BiasAdd/ReadVariableOp2V
)simplecnn/dense_out/MatMul/ReadVariableOp)simplecnn/dense_out/MatMul/ReadVariableOp2b
/simplecnn/fc_batchnorm/batchnorm/ReadVariableOp/simplecnn/fc_batchnorm/batchnorm/ReadVariableOp2f
1simplecnn/fc_batchnorm/batchnorm/ReadVariableOp_11simplecnn/fc_batchnorm/batchnorm/ReadVariableOp_12f
1simplecnn/fc_batchnorm/batchnorm/ReadVariableOp_21simplecnn/fc_batchnorm/batchnorm/ReadVariableOp_22j
3simplecnn/fc_batchnorm/batchnorm/mul/ReadVariableOp3simplecnn/fc_batchnorm/batchnorm/mul/ReadVariableOp2F
!simplecnn/fc_prelu/ReadVariableOp!simplecnn/fc_prelu/ReadVariableOp2d
0simplecnn/featurewise_std/Reshape/ReadVariableOp0simplecnn/featurewise_std/Reshape/ReadVariableOp2h
2simplecnn/featurewise_std/Reshape_1/ReadVariableOp2simplecnn/featurewise_std/Reshape_1/ReadVariableOp:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
"
_user_specified_name
input_sp
¼2
±
D__inference_blocks/b2_layer_call_and_return_conditional_losses_52169

inputsG
-blocks_b2_conv_conv2d_readvariableop_resource:<
.blocks_b2_conv_biasadd_readvariableop_resource:9
+blocks_b2_batchnorm_readvariableop_resource:;
-blocks_b2_batchnorm_readvariableop_1_resource:J
<blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_resource:L
>blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_1_resource:=
'blocks_b2_prelu_readvariableop_resource:0
identity¢"blocks/b2/batchnorm/AssignNewValue¢$blocks/b2/batchnorm/AssignNewValue_1¢3blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp¢5blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1¢"blocks/b2/batchnorm/ReadVariableOp¢$blocks/b2/batchnorm/ReadVariableOp_1¢%blocks/b2/conv/BiasAdd/ReadVariableOp¢$blocks/b2/conv/Conv2D/ReadVariableOp¢blocks/b2/prelu/ReadVariableOpÂ
$blocks/b2/conv/Conv2D/ReadVariableOpReadVariableOp-blocks_b2_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$blocks/b2/conv/Conv2D/ReadVariableOpÐ
blocks/b2/conv/Conv2DConv2Dinputs,blocks/b2/conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*
paddingSAME*
strides
2
blocks/b2/conv/Conv2D¹
%blocks/b2/conv/BiasAdd/ReadVariableOpReadVariableOp.blocks_b2_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%blocks/b2/conv/BiasAdd/ReadVariableOpÄ
blocks/b2/conv/BiasAddBiasAddblocks/b2/conv/Conv2D:output:0-blocks/b2/conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002
blocks/b2/conv/BiasAdd°
"blocks/b2/batchnorm/ReadVariableOpReadVariableOp+blocks_b2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02$
"blocks/b2/batchnorm/ReadVariableOp¶
$blocks/b2/batchnorm/ReadVariableOp_1ReadVariableOp-blocks_b2_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$blocks/b2/batchnorm/ReadVariableOp_1ã
3blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOpReadVariableOp<blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOpé
5blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1é
$blocks/b2/batchnorm/FusedBatchNormV3FusedBatchNormV3blocks/b2/conv/BiasAdd:output:0*blocks/b2/batchnorm/ReadVariableOp:value:0,blocks/b2/batchnorm/ReadVariableOp_1:value:0;blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp:value:0=blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00:::::*
epsilon%o:*
exponential_avg_factor%
×#<2&
$blocks/b2/batchnorm/FusedBatchNormV3¦
"blocks/b2/batchnorm/AssignNewValueAssignVariableOp<blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_resource1blocks/b2/batchnorm/FusedBatchNormV3:batch_mean:04^blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"blocks/b2/batchnorm/AssignNewValue²
$blocks/b2/batchnorm/AssignNewValue_1AssignVariableOp>blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_1_resource5blocks/b2/batchnorm/FusedBatchNormV3:batch_variance:06^blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$blocks/b2/batchnorm/AssignNewValue_1
blocks/b2/prelu/ReluRelu(blocks/b2/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002
blocks/b2/prelu/Relu¬
blocks/b2/prelu/ReadVariableOpReadVariableOp'blocks_b2_prelu_readvariableop_resource*"
_output_shapes
:0*
dtype02 
blocks/b2/prelu/ReadVariableOp
blocks/b2/prelu/NegNeg&blocks/b2/prelu/ReadVariableOp:value:0*
T0*"
_output_shapes
:02
blocks/b2/prelu/Neg
blocks/b2/prelu/Neg_1Neg(blocks/b2/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002
blocks/b2/prelu/Neg_1
blocks/b2/prelu/Relu_1Relublocks/b2/prelu/Neg_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002
blocks/b2/prelu/Relu_1ª
blocks/b2/prelu/mulMulblocks/b2/prelu/Neg:y:0$blocks/b2/prelu/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002
blocks/b2/prelu/mulª
blocks/b2/prelu/addAddV2"blocks/b2/prelu/Relu:activations:0blocks/b2/prelu/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002
blocks/b2/prelu/addÆ
blocks/b2/maxpool/MaxPoolMaxPoolblocks/b2/prelu/add:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2
blocks/b2/maxpool/MaxPoolô
IdentityIdentity"blocks/b2/maxpool/MaxPool:output:0#^blocks/b2/batchnorm/AssignNewValue%^blocks/b2/batchnorm/AssignNewValue_14^blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp6^blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1#^blocks/b2/batchnorm/ReadVariableOp%^blocks/b2/batchnorm/ReadVariableOp_1&^blocks/b2/conv/BiasAdd/ReadVariableOp%^blocks/b2/conv/Conv2D/ReadVariableOp^blocks/b2/prelu/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ00: : : : : : : 2H
"blocks/b2/batchnorm/AssignNewValue"blocks/b2/batchnorm/AssignNewValue2L
$blocks/b2/batchnorm/AssignNewValue_1$blocks/b2/batchnorm/AssignNewValue_12j
3blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp3blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp2n
5blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_15blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_12H
"blocks/b2/batchnorm/ReadVariableOp"blocks/b2/batchnorm/ReadVariableOp2L
$blocks/b2/batchnorm/ReadVariableOp_1$blocks/b2/batchnorm/ReadVariableOp_12N
%blocks/b2/conv/BiasAdd/ReadVariableOp%blocks/b2/conv/BiasAdd/ReadVariableOp2L
$blocks/b2/conv/Conv2D/ReadVariableOp$blocks/b2/conv/Conv2D/ReadVariableOp2@
blocks/b2/prelu/ReadVariableOpblocks/b2/prelu/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
 
_user_specified_nameinputs
À	
±
)__inference_blocks/b2_layer_call_fn_49184
blocks_b2_conv_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:0
identity¢StatefulPartitionedCallÎ
StatefulPartitionedCallStatefulPartitionedCallblocks_b2_conv_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_blocks/b2_layer_call_and_return_conditional_losses_491672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ00: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
.
_user_specified_nameblocks/b2/conv_input
ð
Ç
,__inference_fc_batchnorm_layer_call_fn_51856

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_fc_batchnorm_layer_call_and_return_conditional_losses_501922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Î
3__inference_blocks/b3/batchnorm_layer_call_fn_52617

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_blocks/b3/batchnorm_layer_call_and_return_conditional_losses_495602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê

/__inference_blocks/b2/prelu_layer_call_fn_49100

inputs
unknown:0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ0ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_blocks/b2/prelu_layer_call_and_return_conditional_losses_490922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ0ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:"ÿÿÿÿÿÿÿÿÿ0ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:` \
8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ0ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê

/__inference_blocks/b1/prelu_layer_call_fn_48692

inputs
unknown:`
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ`ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_blocks/b1/prelu_layer_call_and_return_conditional_losses_486842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ`ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:"ÿÿÿÿÿÿÿÿÿ`ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:` \
8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿ`ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é

D__inference_blocks/b3_layer_call_and_return_conditional_losses_49689

inputs.
blocks_b3_conv_49670:"
blocks_b3_conv_49672:'
blocks_b3_batchnorm_49675:'
blocks_b3_batchnorm_49677:'
blocks_b3_batchnorm_49679:'
blocks_b3_batchnorm_49681:+
blocks_b3_prelu_49684:
identity¢+blocks/b3/batchnorm/StatefulPartitionedCall¢&blocks/b3/conv/StatefulPartitionedCall¢'blocks/b3/prelu/StatefulPartitionedCallº
&blocks/b3/conv/StatefulPartitionedCallStatefulPartitionedCallinputsblocks_b3_conv_49670blocks_b3_conv_49672*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_blocks/b3/conv_layer_call_and_return_conditional_losses_495372(
&blocks/b3/conv/StatefulPartitionedCall´
+blocks/b3/batchnorm/StatefulPartitionedCallStatefulPartitionedCall/blocks/b3/conv/StatefulPartitionedCall:output:0blocks_b3_batchnorm_49675blocks_b3_batchnorm_49677blocks_b3_batchnorm_49679blocks_b3_batchnorm_49681*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_blocks/b3/batchnorm_layer_call_and_return_conditional_losses_496252-
+blocks/b3/batchnorm/StatefulPartitionedCallÔ
'blocks/b3/prelu/StatefulPartitionedCallStatefulPartitionedCall4blocks/b3/batchnorm/StatefulPartitionedCall:output:0blocks_b3_prelu_49684*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_blocks/b3/prelu_layer_call_and_return_conditional_losses_495002)
'blocks/b3/prelu/StatefulPartitionedCall£
!blocks/b3/maxpool/PartitionedCallPartitionedCall0blocks/b3/prelu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_blocks/b3/maxpool_layer_call_and_return_conditional_losses_495142#
!blocks/b3/maxpool/PartitionedCall
IdentityIdentity*blocks/b3/maxpool/PartitionedCall:output:0,^blocks/b3/batchnorm/StatefulPartitionedCall'^blocks/b3/conv/StatefulPartitionedCall(^blocks/b3/prelu/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : 2Z
+blocks/b3/batchnorm/StatefulPartitionedCall+blocks/b3/batchnorm/StatefulPartitionedCall2P
&blocks/b3/conv/StatefulPartitionedCall&blocks/b3/conv/StatefulPartitionedCall2R
'blocks/b3/prelu/StatefulPartitionedCall'blocks/b3/prelu/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

a
E__inference_fc_dropout_layer_call_and_return_conditional_losses_50558

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¶
&__inference_blocks_layer_call_fn_50057
blocks_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:`#
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11: 

unknown_12:0$

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18: 

unknown_19:
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallblocks_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19*!
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_blocks_layer_call_and_return_conditional_losses_499652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
&
_user_specified_nameblocks/input


D__inference_blocks/b3_layer_call_and_return_conditional_losses_49769
blocks_b3_conv_input.
blocks_b3_conv_49750:"
blocks_b3_conv_49752:'
blocks_b3_batchnorm_49755:'
blocks_b3_batchnorm_49757:'
blocks_b3_batchnorm_49759:'
blocks_b3_batchnorm_49761:+
blocks_b3_prelu_49764:
identity¢+blocks/b3/batchnorm/StatefulPartitionedCall¢&blocks/b3/conv/StatefulPartitionedCall¢'blocks/b3/prelu/StatefulPartitionedCallÈ
&blocks/b3/conv/StatefulPartitionedCallStatefulPartitionedCallblocks_b3_conv_inputblocks_b3_conv_49750blocks_b3_conv_49752*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_blocks/b3/conv_layer_call_and_return_conditional_losses_495372(
&blocks/b3/conv/StatefulPartitionedCall´
+blocks/b3/batchnorm/StatefulPartitionedCallStatefulPartitionedCall/blocks/b3/conv/StatefulPartitionedCall:output:0blocks_b3_batchnorm_49755blocks_b3_batchnorm_49757blocks_b3_batchnorm_49759blocks_b3_batchnorm_49761*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_blocks/b3/batchnorm_layer_call_and_return_conditional_losses_496252-
+blocks/b3/batchnorm/StatefulPartitionedCallÔ
'blocks/b3/prelu/StatefulPartitionedCallStatefulPartitionedCall4blocks/b3/batchnorm/StatefulPartitionedCall:output:0blocks_b3_prelu_49764*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_blocks/b3/prelu_layer_call_and_return_conditional_losses_495002)
'blocks/b3/prelu/StatefulPartitionedCall£
!blocks/b3/maxpool/PartitionedCallPartitionedCall0blocks/b3/prelu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_blocks/b3/maxpool_layer_call_and_return_conditional_losses_495142#
!blocks/b3/maxpool/PartitionedCall
IdentityIdentity*blocks/b3/maxpool/PartitionedCall:output:0,^blocks/b3/batchnorm/StatefulPartitionedCall'^blocks/b3/conv/StatefulPartitionedCall(^blocks/b3/prelu/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : 2Z
+blocks/b3/batchnorm/StatefulPartitionedCall+blocks/b3/batchnorm/StatefulPartitionedCall2P
&blocks/b3/conv/StatefulPartitionedCall&blocks/b3/conv/StatefulPartitionedCall2R
'blocks/b3/prelu/StatefulPartitionedCall'blocks/b3/prelu/StatefulPartitionedCall:e a
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.
_user_specified_nameblocks/b3/conv_input


N__inference_blocks/b2/batchnorm_layer_call_and_return_conditional_losses_52505

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô
½
N__inference_blocks/b2/batchnorm_layer_call_and_return_conditional_losses_52559

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1þ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ00: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
 
_user_specified_nameinputs

Î
3__inference_blocks/b3/batchnorm_layer_call_fn_52630

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_blocks/b3/batchnorm_layer_call_and_return_conditional_losses_496252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
½
N__inference_blocks/b2/batchnorm_layer_call_and_return_conditional_losses_52523

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò
c
E__inference_fc_dropout_layer_call_and_return_conditional_losses_50446

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
h
L__inference_blocks/b3/maxpool_layer_call_and_return_conditional_losses_49514

inputs
identity¬
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
ª
A__inference_blocks_layer_call_and_return_conditional_losses_51824

inputsQ
7blocks_b1_blocks_b1_conv_conv2d_readvariableop_resource:F
8blocks_b1_blocks_b1_conv_biasadd_readvariableop_resource:C
5blocks_b1_blocks_b1_batchnorm_readvariableop_resource:E
7blocks_b1_blocks_b1_batchnorm_readvariableop_1_resource:T
Fblocks_b1_blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_resource:V
Hblocks_b1_blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_1_resource:G
1blocks_b1_blocks_b1_prelu_readvariableop_resource:`Q
7blocks_b2_blocks_b2_conv_conv2d_readvariableop_resource:F
8blocks_b2_blocks_b2_conv_biasadd_readvariableop_resource:C
5blocks_b2_blocks_b2_batchnorm_readvariableop_resource:E
7blocks_b2_blocks_b2_batchnorm_readvariableop_1_resource:T
Fblocks_b2_blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_resource:V
Hblocks_b2_blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_1_resource:G
1blocks_b2_blocks_b2_prelu_readvariableop_resource:0Q
7blocks_b3_blocks_b3_conv_conv2d_readvariableop_resource:F
8blocks_b3_blocks_b3_conv_biasadd_readvariableop_resource:C
5blocks_b3_blocks_b3_batchnorm_readvariableop_resource:E
7blocks_b3_blocks_b3_batchnorm_readvariableop_1_resource:T
Fblocks_b3_blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_resource:V
Hblocks_b3_blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_1_resource:G
1blocks_b3_blocks_b3_prelu_readvariableop_resource:
identity¢,blocks/b1/blocks/b1/batchnorm/AssignNewValue¢.blocks/b1/blocks/b1/batchnorm/AssignNewValue_1¢=blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp¢?blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1¢,blocks/b1/blocks/b1/batchnorm/ReadVariableOp¢.blocks/b1/blocks/b1/batchnorm/ReadVariableOp_1¢/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp¢.blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOp¢(blocks/b1/blocks/b1/prelu/ReadVariableOp¢,blocks/b2/blocks/b2/batchnorm/AssignNewValue¢.blocks/b2/blocks/b2/batchnorm/AssignNewValue_1¢=blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp¢?blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1¢,blocks/b2/blocks/b2/batchnorm/ReadVariableOp¢.blocks/b2/blocks/b2/batchnorm/ReadVariableOp_1¢/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp¢.blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp¢(blocks/b2/blocks/b2/prelu/ReadVariableOp¢,blocks/b3/blocks/b3/batchnorm/AssignNewValue¢.blocks/b3/blocks/b3/batchnorm/AssignNewValue_1¢=blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp¢?blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1¢,blocks/b3/blocks/b3/batchnorm/ReadVariableOp¢.blocks/b3/blocks/b3/batchnorm/ReadVariableOp_1¢/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp¢.blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp¢(blocks/b3/blocks/b3/prelu/ReadVariableOpà
.blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOpReadVariableOp7blocks_b1_blocks_b1_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOpî
blocks/b1/blocks/b1/conv/Conv2DConv2Dinputs6blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*
paddingSAME*
strides
2!
blocks/b1/blocks/b1/conv/Conv2D×
/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOpReadVariableOp8blocks_b1_blocks_b1_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOpì
 blocks/b1/blocks/b1/conv/BiasAddBiasAdd(blocks/b1/blocks/b1/conv/Conv2D:output:07blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2"
 blocks/b1/blocks/b1/conv/BiasAddÎ
,blocks/b1/blocks/b1/batchnorm/ReadVariableOpReadVariableOp5blocks_b1_blocks_b1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02.
,blocks/b1/blocks/b1/batchnorm/ReadVariableOpÔ
.blocks/b1/blocks/b1/batchnorm/ReadVariableOp_1ReadVariableOp7blocks_b1_blocks_b1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype020
.blocks/b1/blocks/b1/batchnorm/ReadVariableOp_1
=blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOpReadVariableOpFblocks_b1_blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02?
=blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp
?blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHblocks_b1_blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1¯
.blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3FusedBatchNormV3)blocks/b1/blocks/b1/conv/BiasAdd:output:04blocks/b1/blocks/b1/batchnorm/ReadVariableOp:value:06blocks/b1/blocks/b1/batchnorm/ReadVariableOp_1:value:0Eblocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp:value:0Gblocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ``:::::*
epsilon%o:*
exponential_avg_factor%
×#<20
.blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3Ø
,blocks/b1/blocks/b1/batchnorm/AssignNewValueAssignVariableOpFblocks_b1_blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_resource;blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3:batch_mean:0>^blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02.
,blocks/b1/blocks/b1/batchnorm/AssignNewValueä
.blocks/b1/blocks/b1/batchnorm/AssignNewValue_1AssignVariableOpHblocks_b1_blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_1_resource?blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3:batch_variance:0@^blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype020
.blocks/b1/blocks/b1/batchnorm/AssignNewValue_1¶
blocks/b1/blocks/b1/prelu/ReluRelu2blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2 
blocks/b1/blocks/b1/prelu/ReluÊ
(blocks/b1/blocks/b1/prelu/ReadVariableOpReadVariableOp1blocks_b1_blocks_b1_prelu_readvariableop_resource*"
_output_shapes
:`*
dtype02*
(blocks/b1/blocks/b1/prelu/ReadVariableOp¤
blocks/b1/blocks/b1/prelu/NegNeg0blocks/b1/blocks/b1/prelu/ReadVariableOp:value:0*
T0*"
_output_shapes
:`2
blocks/b1/blocks/b1/prelu/Neg·
blocks/b1/blocks/b1/prelu/Neg_1Neg2blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2!
blocks/b1/blocks/b1/prelu/Neg_1«
 blocks/b1/blocks/b1/prelu/Relu_1Relu#blocks/b1/blocks/b1/prelu/Neg_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2"
 blocks/b1/blocks/b1/prelu/Relu_1Ò
blocks/b1/blocks/b1/prelu/mulMul!blocks/b1/blocks/b1/prelu/Neg:y:0.blocks/b1/blocks/b1/prelu/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
blocks/b1/blocks/b1/prelu/mulÒ
blocks/b1/blocks/b1/prelu/addAddV2,blocks/b1/blocks/b1/prelu/Relu:activations:0!blocks/b1/blocks/b1/prelu/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
blocks/b1/blocks/b1/prelu/addä
#blocks/b1/blocks/b1/maxpool/MaxPoolMaxPool!blocks/b1/blocks/b1/prelu/add:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*
ksize
*
paddingSAME*
strides
2%
#blocks/b1/blocks/b1/maxpool/MaxPoolà
.blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOpReadVariableOp7blocks_b2_blocks_b2_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp
blocks/b2/blocks/b2/conv/Conv2DConv2D,blocks/b1/blocks/b1/maxpool/MaxPool:output:06blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*
paddingSAME*
strides
2!
blocks/b2/blocks/b2/conv/Conv2D×
/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOpReadVariableOp8blocks_b2_blocks_b2_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOpì
 blocks/b2/blocks/b2/conv/BiasAddBiasAdd(blocks/b2/blocks/b2/conv/Conv2D:output:07blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002"
 blocks/b2/blocks/b2/conv/BiasAddÎ
,blocks/b2/blocks/b2/batchnorm/ReadVariableOpReadVariableOp5blocks_b2_blocks_b2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02.
,blocks/b2/blocks/b2/batchnorm/ReadVariableOpÔ
.blocks/b2/blocks/b2/batchnorm/ReadVariableOp_1ReadVariableOp7blocks_b2_blocks_b2_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype020
.blocks/b2/blocks/b2/batchnorm/ReadVariableOp_1
=blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOpReadVariableOpFblocks_b2_blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02?
=blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp
?blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHblocks_b2_blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1¯
.blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3FusedBatchNormV3)blocks/b2/blocks/b2/conv/BiasAdd:output:04blocks/b2/blocks/b2/batchnorm/ReadVariableOp:value:06blocks/b2/blocks/b2/batchnorm/ReadVariableOp_1:value:0Eblocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp:value:0Gblocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00:::::*
epsilon%o:*
exponential_avg_factor%
×#<20
.blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3Ø
,blocks/b2/blocks/b2/batchnorm/AssignNewValueAssignVariableOpFblocks_b2_blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_resource;blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3:batch_mean:0>^blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02.
,blocks/b2/blocks/b2/batchnorm/AssignNewValueä
.blocks/b2/blocks/b2/batchnorm/AssignNewValue_1AssignVariableOpHblocks_b2_blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_1_resource?blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3:batch_variance:0@^blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype020
.blocks/b2/blocks/b2/batchnorm/AssignNewValue_1¶
blocks/b2/blocks/b2/prelu/ReluRelu2blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002 
blocks/b2/blocks/b2/prelu/ReluÊ
(blocks/b2/blocks/b2/prelu/ReadVariableOpReadVariableOp1blocks_b2_blocks_b2_prelu_readvariableop_resource*"
_output_shapes
:0*
dtype02*
(blocks/b2/blocks/b2/prelu/ReadVariableOp¤
blocks/b2/blocks/b2/prelu/NegNeg0blocks/b2/blocks/b2/prelu/ReadVariableOp:value:0*
T0*"
_output_shapes
:02
blocks/b2/blocks/b2/prelu/Neg·
blocks/b2/blocks/b2/prelu/Neg_1Neg2blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002!
blocks/b2/blocks/b2/prelu/Neg_1«
 blocks/b2/blocks/b2/prelu/Relu_1Relu#blocks/b2/blocks/b2/prelu/Neg_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002"
 blocks/b2/blocks/b2/prelu/Relu_1Ò
blocks/b2/blocks/b2/prelu/mulMul!blocks/b2/blocks/b2/prelu/Neg:y:0.blocks/b2/blocks/b2/prelu/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002
blocks/b2/blocks/b2/prelu/mulÒ
blocks/b2/blocks/b2/prelu/addAddV2,blocks/b2/blocks/b2/prelu/Relu:activations:0!blocks/b2/blocks/b2/prelu/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002
blocks/b2/blocks/b2/prelu/addä
#blocks/b2/blocks/b2/maxpool/MaxPoolMaxPool!blocks/b2/blocks/b2/prelu/add:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2%
#blocks/b2/blocks/b2/maxpool/MaxPoolà
.blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOpReadVariableOp7blocks_b3_blocks_b3_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp
blocks/b3/blocks/b3/conv/Conv2DConv2D,blocks/b2/blocks/b2/maxpool/MaxPool:output:06blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2!
blocks/b3/blocks/b3/conv/Conv2D×
/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOpReadVariableOp8blocks_b3_blocks_b3_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOpì
 blocks/b3/blocks/b3/conv/BiasAddBiasAdd(blocks/b3/blocks/b3/conv/Conv2D:output:07blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 blocks/b3/blocks/b3/conv/BiasAddÎ
,blocks/b3/blocks/b3/batchnorm/ReadVariableOpReadVariableOp5blocks_b3_blocks_b3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02.
,blocks/b3/blocks/b3/batchnorm/ReadVariableOpÔ
.blocks/b3/blocks/b3/batchnorm/ReadVariableOp_1ReadVariableOp7blocks_b3_blocks_b3_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype020
.blocks/b3/blocks/b3/batchnorm/ReadVariableOp_1
=blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOpReadVariableOpFblocks_b3_blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02?
=blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp
?blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHblocks_b3_blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1¯
.blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3FusedBatchNormV3)blocks/b3/blocks/b3/conv/BiasAdd:output:04blocks/b3/blocks/b3/batchnorm/ReadVariableOp:value:06blocks/b3/blocks/b3/batchnorm/ReadVariableOp_1:value:0Eblocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp:value:0Gblocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<20
.blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3Ø
,blocks/b3/blocks/b3/batchnorm/AssignNewValueAssignVariableOpFblocks_b3_blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_resource;blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3:batch_mean:0>^blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02.
,blocks/b3/blocks/b3/batchnorm/AssignNewValueä
.blocks/b3/blocks/b3/batchnorm/AssignNewValue_1AssignVariableOpHblocks_b3_blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_1_resource?blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3:batch_variance:0@^blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype020
.blocks/b3/blocks/b3/batchnorm/AssignNewValue_1¶
blocks/b3/blocks/b3/prelu/ReluRelu2blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
blocks/b3/blocks/b3/prelu/ReluÊ
(blocks/b3/blocks/b3/prelu/ReadVariableOpReadVariableOp1blocks_b3_blocks_b3_prelu_readvariableop_resource*"
_output_shapes
:*
dtype02*
(blocks/b3/blocks/b3/prelu/ReadVariableOp¤
blocks/b3/blocks/b3/prelu/NegNeg0blocks/b3/blocks/b3/prelu/ReadVariableOp:value:0*
T0*"
_output_shapes
:2
blocks/b3/blocks/b3/prelu/Neg·
blocks/b3/blocks/b3/prelu/Neg_1Neg2blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
blocks/b3/blocks/b3/prelu/Neg_1«
 blocks/b3/blocks/b3/prelu/Relu_1Relu#blocks/b3/blocks/b3/prelu/Neg_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 blocks/b3/blocks/b3/prelu/Relu_1Ò
blocks/b3/blocks/b3/prelu/mulMul!blocks/b3/blocks/b3/prelu/Neg:y:0.blocks/b3/blocks/b3/prelu/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
blocks/b3/blocks/b3/prelu/mulÒ
blocks/b3/blocks/b3/prelu/addAddV2,blocks/b3/blocks/b3/prelu/Relu:activations:0!blocks/b3/blocks/b3/prelu/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
blocks/b3/blocks/b3/prelu/addä
#blocks/b3/blocks/b3/maxpool/MaxPoolMaxPool!blocks/b3/blocks/b3/prelu/add:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2%
#blocks/b3/blocks/b3/maxpool/MaxPoolø
IdentityIdentity,blocks/b3/blocks/b3/maxpool/MaxPool:output:0-^blocks/b1/blocks/b1/batchnorm/AssignNewValue/^blocks/b1/blocks/b1/batchnorm/AssignNewValue_1>^blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp@^blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1-^blocks/b1/blocks/b1/batchnorm/ReadVariableOp/^blocks/b1/blocks/b1/batchnorm/ReadVariableOp_10^blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp/^blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOp)^blocks/b1/blocks/b1/prelu/ReadVariableOp-^blocks/b2/blocks/b2/batchnorm/AssignNewValue/^blocks/b2/blocks/b2/batchnorm/AssignNewValue_1>^blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp@^blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1-^blocks/b2/blocks/b2/batchnorm/ReadVariableOp/^blocks/b2/blocks/b2/batchnorm/ReadVariableOp_10^blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp/^blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp)^blocks/b2/blocks/b2/prelu/ReadVariableOp-^blocks/b3/blocks/b3/batchnorm/AssignNewValue/^blocks/b3/blocks/b3/batchnorm/AssignNewValue_1>^blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp@^blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1-^blocks/b3/blocks/b3/batchnorm/ReadVariableOp/^blocks/b3/blocks/b3/batchnorm/ReadVariableOp_10^blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp/^blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp)^blocks/b3/blocks/b3/prelu/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : 2\
,blocks/b1/blocks/b1/batchnorm/AssignNewValue,blocks/b1/blocks/b1/batchnorm/AssignNewValue2`
.blocks/b1/blocks/b1/batchnorm/AssignNewValue_1.blocks/b1/blocks/b1/batchnorm/AssignNewValue_12~
=blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp=blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp2
?blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1?blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_12\
,blocks/b1/blocks/b1/batchnorm/ReadVariableOp,blocks/b1/blocks/b1/batchnorm/ReadVariableOp2`
.blocks/b1/blocks/b1/batchnorm/ReadVariableOp_1.blocks/b1/blocks/b1/batchnorm/ReadVariableOp_12b
/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp2`
.blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOp.blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOp2T
(blocks/b1/blocks/b1/prelu/ReadVariableOp(blocks/b1/blocks/b1/prelu/ReadVariableOp2\
,blocks/b2/blocks/b2/batchnorm/AssignNewValue,blocks/b2/blocks/b2/batchnorm/AssignNewValue2`
.blocks/b2/blocks/b2/batchnorm/AssignNewValue_1.blocks/b2/blocks/b2/batchnorm/AssignNewValue_12~
=blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp=blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp2
?blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1?blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_12\
,blocks/b2/blocks/b2/batchnorm/ReadVariableOp,blocks/b2/blocks/b2/batchnorm/ReadVariableOp2`
.blocks/b2/blocks/b2/batchnorm/ReadVariableOp_1.blocks/b2/blocks/b2/batchnorm/ReadVariableOp_12b
/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp2`
.blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp.blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp2T
(blocks/b2/blocks/b2/prelu/ReadVariableOp(blocks/b2/blocks/b2/prelu/ReadVariableOp2\
,blocks/b3/blocks/b3/batchnorm/AssignNewValue,blocks/b3/blocks/b3/batchnorm/AssignNewValue2`
.blocks/b3/blocks/b3/batchnorm/AssignNewValue_1.blocks/b3/blocks/b3/batchnorm/AssignNewValue_12~
=blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp=blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp2
?blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1?blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_12\
,blocks/b3/blocks/b3/batchnorm/ReadVariableOp,blocks/b3/blocks/b3/batchnorm/ReadVariableOp2`
.blocks/b3/blocks/b3/batchnorm/ReadVariableOp_1.blocks/b3/blocks/b3/batchnorm/ReadVariableOp_12b
/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp2`
.blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp.blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp2T
(blocks/b3/blocks/b3/prelu/ReadVariableOp(blocks/b3/blocks/b3/prelu/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs

Î
3__inference_blocks/b1/batchnorm_layer_call_fn_52344

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_blocks/b1/batchnorm_layer_call_and_return_conditional_losses_488092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ``: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
ò
c
E__inference_fc_dropout_layer_call_and_return_conditional_losses_51938

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾	
±
)__inference_blocks/b1_layer_call_fn_48909
blocks_b1_conv_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:`
identity¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallblocks_b1_conv_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_blocks/b1_layer_call_and_return_conditional_losses_488732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ``: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
.
_user_specified_nameblocks/b1/conv_input


N__inference_blocks/b2/batchnorm_layer_call_and_return_conditional_losses_48975

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª

A__inference_blocks_layer_call_and_return_conditional_losses_51733

inputsQ
7blocks_b1_blocks_b1_conv_conv2d_readvariableop_resource:F
8blocks_b1_blocks_b1_conv_biasadd_readvariableop_resource:C
5blocks_b1_blocks_b1_batchnorm_readvariableop_resource:E
7blocks_b1_blocks_b1_batchnorm_readvariableop_1_resource:T
Fblocks_b1_blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_resource:V
Hblocks_b1_blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_1_resource:G
1blocks_b1_blocks_b1_prelu_readvariableop_resource:`Q
7blocks_b2_blocks_b2_conv_conv2d_readvariableop_resource:F
8blocks_b2_blocks_b2_conv_biasadd_readvariableop_resource:C
5blocks_b2_blocks_b2_batchnorm_readvariableop_resource:E
7blocks_b2_blocks_b2_batchnorm_readvariableop_1_resource:T
Fblocks_b2_blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_resource:V
Hblocks_b2_blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_1_resource:G
1blocks_b2_blocks_b2_prelu_readvariableop_resource:0Q
7blocks_b3_blocks_b3_conv_conv2d_readvariableop_resource:F
8blocks_b3_blocks_b3_conv_biasadd_readvariableop_resource:C
5blocks_b3_blocks_b3_batchnorm_readvariableop_resource:E
7blocks_b3_blocks_b3_batchnorm_readvariableop_1_resource:T
Fblocks_b3_blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_resource:V
Hblocks_b3_blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_1_resource:G
1blocks_b3_blocks_b3_prelu_readvariableop_resource:
identity¢=blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp¢?blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1¢,blocks/b1/blocks/b1/batchnorm/ReadVariableOp¢.blocks/b1/blocks/b1/batchnorm/ReadVariableOp_1¢/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp¢.blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOp¢(blocks/b1/blocks/b1/prelu/ReadVariableOp¢=blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp¢?blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1¢,blocks/b2/blocks/b2/batchnorm/ReadVariableOp¢.blocks/b2/blocks/b2/batchnorm/ReadVariableOp_1¢/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp¢.blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp¢(blocks/b2/blocks/b2/prelu/ReadVariableOp¢=blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp¢?blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1¢,blocks/b3/blocks/b3/batchnorm/ReadVariableOp¢.blocks/b3/blocks/b3/batchnorm/ReadVariableOp_1¢/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp¢.blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp¢(blocks/b3/blocks/b3/prelu/ReadVariableOpà
.blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOpReadVariableOp7blocks_b1_blocks_b1_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOpî
blocks/b1/blocks/b1/conv/Conv2DConv2Dinputs6blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*
paddingSAME*
strides
2!
blocks/b1/blocks/b1/conv/Conv2D×
/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOpReadVariableOp8blocks_b1_blocks_b1_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOpì
 blocks/b1/blocks/b1/conv/BiasAddBiasAdd(blocks/b1/blocks/b1/conv/Conv2D:output:07blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2"
 blocks/b1/blocks/b1/conv/BiasAddÎ
,blocks/b1/blocks/b1/batchnorm/ReadVariableOpReadVariableOp5blocks_b1_blocks_b1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02.
,blocks/b1/blocks/b1/batchnorm/ReadVariableOpÔ
.blocks/b1/blocks/b1/batchnorm/ReadVariableOp_1ReadVariableOp7blocks_b1_blocks_b1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype020
.blocks/b1/blocks/b1/batchnorm/ReadVariableOp_1
=blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOpReadVariableOpFblocks_b1_blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02?
=blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp
?blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHblocks_b1_blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1¡
.blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3FusedBatchNormV3)blocks/b1/blocks/b1/conv/BiasAdd:output:04blocks/b1/blocks/b1/batchnorm/ReadVariableOp:value:06blocks/b1/blocks/b1/batchnorm/ReadVariableOp_1:value:0Eblocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp:value:0Gblocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ``:::::*
epsilon%o:*
is_training( 20
.blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3¶
blocks/b1/blocks/b1/prelu/ReluRelu2blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2 
blocks/b1/blocks/b1/prelu/ReluÊ
(blocks/b1/blocks/b1/prelu/ReadVariableOpReadVariableOp1blocks_b1_blocks_b1_prelu_readvariableop_resource*"
_output_shapes
:`*
dtype02*
(blocks/b1/blocks/b1/prelu/ReadVariableOp¤
blocks/b1/blocks/b1/prelu/NegNeg0blocks/b1/blocks/b1/prelu/ReadVariableOp:value:0*
T0*"
_output_shapes
:`2
blocks/b1/blocks/b1/prelu/Neg·
blocks/b1/blocks/b1/prelu/Neg_1Neg2blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2!
blocks/b1/blocks/b1/prelu/Neg_1«
 blocks/b1/blocks/b1/prelu/Relu_1Relu#blocks/b1/blocks/b1/prelu/Neg_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2"
 blocks/b1/blocks/b1/prelu/Relu_1Ò
blocks/b1/blocks/b1/prelu/mulMul!blocks/b1/blocks/b1/prelu/Neg:y:0.blocks/b1/blocks/b1/prelu/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
blocks/b1/blocks/b1/prelu/mulÒ
blocks/b1/blocks/b1/prelu/addAddV2,blocks/b1/blocks/b1/prelu/Relu:activations:0!blocks/b1/blocks/b1/prelu/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
blocks/b1/blocks/b1/prelu/addä
#blocks/b1/blocks/b1/maxpool/MaxPoolMaxPool!blocks/b1/blocks/b1/prelu/add:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*
ksize
*
paddingSAME*
strides
2%
#blocks/b1/blocks/b1/maxpool/MaxPoolà
.blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOpReadVariableOp7blocks_b2_blocks_b2_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp
blocks/b2/blocks/b2/conv/Conv2DConv2D,blocks/b1/blocks/b1/maxpool/MaxPool:output:06blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*
paddingSAME*
strides
2!
blocks/b2/blocks/b2/conv/Conv2D×
/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOpReadVariableOp8blocks_b2_blocks_b2_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOpì
 blocks/b2/blocks/b2/conv/BiasAddBiasAdd(blocks/b2/blocks/b2/conv/Conv2D:output:07blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002"
 blocks/b2/blocks/b2/conv/BiasAddÎ
,blocks/b2/blocks/b2/batchnorm/ReadVariableOpReadVariableOp5blocks_b2_blocks_b2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02.
,blocks/b2/blocks/b2/batchnorm/ReadVariableOpÔ
.blocks/b2/blocks/b2/batchnorm/ReadVariableOp_1ReadVariableOp7blocks_b2_blocks_b2_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype020
.blocks/b2/blocks/b2/batchnorm/ReadVariableOp_1
=blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOpReadVariableOpFblocks_b2_blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02?
=blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp
?blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHblocks_b2_blocks_b2_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1¡
.blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3FusedBatchNormV3)blocks/b2/blocks/b2/conv/BiasAdd:output:04blocks/b2/blocks/b2/batchnorm/ReadVariableOp:value:06blocks/b2/blocks/b2/batchnorm/ReadVariableOp_1:value:0Eblocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp:value:0Gblocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00:::::*
epsilon%o:*
is_training( 20
.blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3¶
blocks/b2/blocks/b2/prelu/ReluRelu2blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002 
blocks/b2/blocks/b2/prelu/ReluÊ
(blocks/b2/blocks/b2/prelu/ReadVariableOpReadVariableOp1blocks_b2_blocks_b2_prelu_readvariableop_resource*"
_output_shapes
:0*
dtype02*
(blocks/b2/blocks/b2/prelu/ReadVariableOp¤
blocks/b2/blocks/b2/prelu/NegNeg0blocks/b2/blocks/b2/prelu/ReadVariableOp:value:0*
T0*"
_output_shapes
:02
blocks/b2/blocks/b2/prelu/Neg·
blocks/b2/blocks/b2/prelu/Neg_1Neg2blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002!
blocks/b2/blocks/b2/prelu/Neg_1«
 blocks/b2/blocks/b2/prelu/Relu_1Relu#blocks/b2/blocks/b2/prelu/Neg_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002"
 blocks/b2/blocks/b2/prelu/Relu_1Ò
blocks/b2/blocks/b2/prelu/mulMul!blocks/b2/blocks/b2/prelu/Neg:y:0.blocks/b2/blocks/b2/prelu/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002
blocks/b2/blocks/b2/prelu/mulÒ
blocks/b2/blocks/b2/prelu/addAddV2,blocks/b2/blocks/b2/prelu/Relu:activations:0!blocks/b2/blocks/b2/prelu/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002
blocks/b2/blocks/b2/prelu/addä
#blocks/b2/blocks/b2/maxpool/MaxPoolMaxPool!blocks/b2/blocks/b2/prelu/add:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2%
#blocks/b2/blocks/b2/maxpool/MaxPoolà
.blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOpReadVariableOp7blocks_b3_blocks_b3_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp
blocks/b3/blocks/b3/conv/Conv2DConv2D,blocks/b2/blocks/b2/maxpool/MaxPool:output:06blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2!
blocks/b3/blocks/b3/conv/Conv2D×
/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOpReadVariableOp8blocks_b3_blocks_b3_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOpì
 blocks/b3/blocks/b3/conv/BiasAddBiasAdd(blocks/b3/blocks/b3/conv/Conv2D:output:07blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 blocks/b3/blocks/b3/conv/BiasAddÎ
,blocks/b3/blocks/b3/batchnorm/ReadVariableOpReadVariableOp5blocks_b3_blocks_b3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02.
,blocks/b3/blocks/b3/batchnorm/ReadVariableOpÔ
.blocks/b3/blocks/b3/batchnorm/ReadVariableOp_1ReadVariableOp7blocks_b3_blocks_b3_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype020
.blocks/b3/blocks/b3/batchnorm/ReadVariableOp_1
=blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOpReadVariableOpFblocks_b3_blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02?
=blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp
?blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHblocks_b3_blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1¡
.blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3FusedBatchNormV3)blocks/b3/blocks/b3/conv/BiasAdd:output:04blocks/b3/blocks/b3/batchnorm/ReadVariableOp:value:06blocks/b3/blocks/b3/batchnorm/ReadVariableOp_1:value:0Eblocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp:value:0Gblocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 20
.blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3¶
blocks/b3/blocks/b3/prelu/ReluRelu2blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
blocks/b3/blocks/b3/prelu/ReluÊ
(blocks/b3/blocks/b3/prelu/ReadVariableOpReadVariableOp1blocks_b3_blocks_b3_prelu_readvariableop_resource*"
_output_shapes
:*
dtype02*
(blocks/b3/blocks/b3/prelu/ReadVariableOp¤
blocks/b3/blocks/b3/prelu/NegNeg0blocks/b3/blocks/b3/prelu/ReadVariableOp:value:0*
T0*"
_output_shapes
:2
blocks/b3/blocks/b3/prelu/Neg·
blocks/b3/blocks/b3/prelu/Neg_1Neg2blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
blocks/b3/blocks/b3/prelu/Neg_1«
 blocks/b3/blocks/b3/prelu/Relu_1Relu#blocks/b3/blocks/b3/prelu/Neg_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 blocks/b3/blocks/b3/prelu/Relu_1Ò
blocks/b3/blocks/b3/prelu/mulMul!blocks/b3/blocks/b3/prelu/Neg:y:0.blocks/b3/blocks/b3/prelu/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
blocks/b3/blocks/b3/prelu/mulÒ
blocks/b3/blocks/b3/prelu/addAddV2,blocks/b3/blocks/b3/prelu/Relu:activations:0!blocks/b3/blocks/b3/prelu/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
blocks/b3/blocks/b3/prelu/addä
#blocks/b3/blocks/b3/maxpool/MaxPoolMaxPool!blocks/b3/blocks/b3/prelu/add:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2%
#blocks/b3/blocks/b3/maxpool/MaxPoolØ	
IdentityIdentity,blocks/b3/blocks/b3/maxpool/MaxPool:output:0>^blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp@^blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1-^blocks/b1/blocks/b1/batchnorm/ReadVariableOp/^blocks/b1/blocks/b1/batchnorm/ReadVariableOp_10^blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp/^blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOp)^blocks/b1/blocks/b1/prelu/ReadVariableOp>^blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp@^blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1-^blocks/b2/blocks/b2/batchnorm/ReadVariableOp/^blocks/b2/blocks/b2/batchnorm/ReadVariableOp_10^blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp/^blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp)^blocks/b2/blocks/b2/prelu/ReadVariableOp>^blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp@^blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1-^blocks/b3/blocks/b3/batchnorm/ReadVariableOp/^blocks/b3/blocks/b3/batchnorm/ReadVariableOp_10^blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp/^blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp)^blocks/b3/blocks/b3/prelu/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : 2~
=blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp=blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp2
?blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1?blocks/b1/blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_12\
,blocks/b1/blocks/b1/batchnorm/ReadVariableOp,blocks/b1/blocks/b1/batchnorm/ReadVariableOp2`
.blocks/b1/blocks/b1/batchnorm/ReadVariableOp_1.blocks/b1/blocks/b1/batchnorm/ReadVariableOp_12b
/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp/blocks/b1/blocks/b1/conv/BiasAdd/ReadVariableOp2`
.blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOp.blocks/b1/blocks/b1/conv/Conv2D/ReadVariableOp2T
(blocks/b1/blocks/b1/prelu/ReadVariableOp(blocks/b1/blocks/b1/prelu/ReadVariableOp2~
=blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp=blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp2
?blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_1?blocks/b2/blocks/b2/batchnorm/FusedBatchNormV3/ReadVariableOp_12\
,blocks/b2/blocks/b2/batchnorm/ReadVariableOp,blocks/b2/blocks/b2/batchnorm/ReadVariableOp2`
.blocks/b2/blocks/b2/batchnorm/ReadVariableOp_1.blocks/b2/blocks/b2/batchnorm/ReadVariableOp_12b
/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp/blocks/b2/blocks/b2/conv/BiasAdd/ReadVariableOp2`
.blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp.blocks/b2/blocks/b2/conv/Conv2D/ReadVariableOp2T
(blocks/b2/blocks/b2/prelu/ReadVariableOp(blocks/b2/blocks/b2/prelu/ReadVariableOp2~
=blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp=blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp2
?blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1?blocks/b3/blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_12\
,blocks/b3/blocks/b3/batchnorm/ReadVariableOp,blocks/b3/blocks/b3/batchnorm/ReadVariableOp2`
.blocks/b3/blocks/b3/batchnorm/ReadVariableOp_1.blocks/b3/blocks/b3/batchnorm/ReadVariableOp_12b
/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp/blocks/b3/blocks/b3/conv/BiasAdd/ReadVariableOp2`
.blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp.blocks/b3/blocks/b3/conv/Conv2D/ReadVariableOp2T
(blocks/b3/blocks/b3/prelu/ReadVariableOp(blocks/b3/blocks/b3/prelu/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
ô
½
N__inference_blocks/b1/batchnorm_layer_call_and_return_conditional_losses_48809

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ``:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1þ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ``: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
æ
Î
3__inference_blocks/b1/batchnorm_layer_call_fn_52305

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_blocks/b1/batchnorm_layer_call_and_return_conditional_losses_485672
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 +
å
D__inference_blocks/b1_layer_call_and_return_conditional_losses_52032

inputsG
-blocks_b1_conv_conv2d_readvariableop_resource:<
.blocks_b1_conv_biasadd_readvariableop_resource:9
+blocks_b1_batchnorm_readvariableop_resource:;
-blocks_b1_batchnorm_readvariableop_1_resource:J
<blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_resource:L
>blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_1_resource:=
'blocks_b1_prelu_readvariableop_resource:`
identity¢3blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp¢5blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1¢"blocks/b1/batchnorm/ReadVariableOp¢$blocks/b1/batchnorm/ReadVariableOp_1¢%blocks/b1/conv/BiasAdd/ReadVariableOp¢$blocks/b1/conv/Conv2D/ReadVariableOp¢blocks/b1/prelu/ReadVariableOpÂ
$blocks/b1/conv/Conv2D/ReadVariableOpReadVariableOp-blocks_b1_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$blocks/b1/conv/Conv2D/ReadVariableOpÐ
blocks/b1/conv/Conv2DConv2Dinputs,blocks/b1/conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*
paddingSAME*
strides
2
blocks/b1/conv/Conv2D¹
%blocks/b1/conv/BiasAdd/ReadVariableOpReadVariableOp.blocks_b1_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%blocks/b1/conv/BiasAdd/ReadVariableOpÄ
blocks/b1/conv/BiasAddBiasAddblocks/b1/conv/Conv2D:output:0-blocks/b1/conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
blocks/b1/conv/BiasAdd°
"blocks/b1/batchnorm/ReadVariableOpReadVariableOp+blocks_b1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02$
"blocks/b1/batchnorm/ReadVariableOp¶
$blocks/b1/batchnorm/ReadVariableOp_1ReadVariableOp-blocks_b1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$blocks/b1/batchnorm/ReadVariableOp_1ã
3blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOpReadVariableOp<blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOpé
5blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>blocks_b1_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1Û
$blocks/b1/batchnorm/FusedBatchNormV3FusedBatchNormV3blocks/b1/conv/BiasAdd:output:0*blocks/b1/batchnorm/ReadVariableOp:value:0,blocks/b1/batchnorm/ReadVariableOp_1:value:0;blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp:value:0=blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ``:::::*
epsilon%o:*
is_training( 2&
$blocks/b1/batchnorm/FusedBatchNormV3
blocks/b1/prelu/ReluRelu(blocks/b1/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
blocks/b1/prelu/Relu¬
blocks/b1/prelu/ReadVariableOpReadVariableOp'blocks_b1_prelu_readvariableop_resource*"
_output_shapes
:`*
dtype02 
blocks/b1/prelu/ReadVariableOp
blocks/b1/prelu/NegNeg&blocks/b1/prelu/ReadVariableOp:value:0*
T0*"
_output_shapes
:`2
blocks/b1/prelu/Neg
blocks/b1/prelu/Neg_1Neg(blocks/b1/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
blocks/b1/prelu/Neg_1
blocks/b1/prelu/Relu_1Relublocks/b1/prelu/Neg_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
blocks/b1/prelu/Relu_1ª
blocks/b1/prelu/mulMulblocks/b1/prelu/Neg:y:0$blocks/b1/prelu/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
blocks/b1/prelu/mulª
blocks/b1/prelu/addAddV2"blocks/b1/prelu/Relu:activations:0blocks/b1/prelu/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
blocks/b1/prelu/addÆ
blocks/b1/maxpool/MaxPoolMaxPoolblocks/b1/prelu/add:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*
ksize
*
paddingSAME*
strides
2
blocks/b1/maxpool/MaxPool¨
IdentityIdentity"blocks/b1/maxpool/MaxPool:output:04^blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp6^blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_1#^blocks/b1/batchnorm/ReadVariableOp%^blocks/b1/batchnorm/ReadVariableOp_1&^blocks/b1/conv/BiasAdd/ReadVariableOp%^blocks/b1/conv/Conv2D/ReadVariableOp^blocks/b1/prelu/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ``: : : : : : : 2j
3blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp3blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp2n
5blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_15blocks/b1/batchnorm/FusedBatchNormV3/ReadVariableOp_12H
"blocks/b1/batchnorm/ReadVariableOp"blocks/b1/batchnorm/ReadVariableOp2L
$blocks/b1/batchnorm/ReadVariableOp_1$blocks/b1/batchnorm/ReadVariableOp_12N
%blocks/b1/conv/BiasAdd/ReadVariableOp%blocks/b1/conv/BiasAdd/ReadVariableOp2L
$blocks/b1/conv/Conv2D/ReadVariableOp$blocks/b1/conv/Conv2D/ReadVariableOp2@
blocks/b1/prelu/ReadVariableOpblocks/b1/prelu/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs

Î
3__inference_blocks/b2/batchnorm_layer_call_fn_52474

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_blocks/b2/batchnorm_layer_call_and_return_conditional_losses_491522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ00: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
 
_user_specified_nameinputs
æ
Î
3__inference_blocks/b3/batchnorm_layer_call_fn_52591

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_blocks/b3/batchnorm_layer_call_and_return_conditional_losses_493832
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼2
±
D__inference_blocks/b3_layer_call_and_return_conditional_losses_52273

inputsG
-blocks_b3_conv_conv2d_readvariableop_resource:<
.blocks_b3_conv_biasadd_readvariableop_resource:9
+blocks_b3_batchnorm_readvariableop_resource:;
-blocks_b3_batchnorm_readvariableop_1_resource:J
<blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_resource:L
>blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_1_resource:=
'blocks_b3_prelu_readvariableop_resource:
identity¢"blocks/b3/batchnorm/AssignNewValue¢$blocks/b3/batchnorm/AssignNewValue_1¢3blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp¢5blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1¢"blocks/b3/batchnorm/ReadVariableOp¢$blocks/b3/batchnorm/ReadVariableOp_1¢%blocks/b3/conv/BiasAdd/ReadVariableOp¢$blocks/b3/conv/Conv2D/ReadVariableOp¢blocks/b3/prelu/ReadVariableOpÂ
$blocks/b3/conv/Conv2D/ReadVariableOpReadVariableOp-blocks_b3_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$blocks/b3/conv/Conv2D/ReadVariableOpÐ
blocks/b3/conv/Conv2DConv2Dinputs,blocks/b3/conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
blocks/b3/conv/Conv2D¹
%blocks/b3/conv/BiasAdd/ReadVariableOpReadVariableOp.blocks_b3_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%blocks/b3/conv/BiasAdd/ReadVariableOpÄ
blocks/b3/conv/BiasAddBiasAddblocks/b3/conv/Conv2D:output:0-blocks/b3/conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
blocks/b3/conv/BiasAdd°
"blocks/b3/batchnorm/ReadVariableOpReadVariableOp+blocks_b3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02$
"blocks/b3/batchnorm/ReadVariableOp¶
$blocks/b3/batchnorm/ReadVariableOp_1ReadVariableOp-blocks_b3_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$blocks/b3/batchnorm/ReadVariableOp_1ã
3blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOpReadVariableOp<blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOpé
5blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1é
$blocks/b3/batchnorm/FusedBatchNormV3FusedBatchNormV3blocks/b3/conv/BiasAdd:output:0*blocks/b3/batchnorm/ReadVariableOp:value:0,blocks/b3/batchnorm/ReadVariableOp_1:value:0;blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp:value:0=blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2&
$blocks/b3/batchnorm/FusedBatchNormV3¦
"blocks/b3/batchnorm/AssignNewValueAssignVariableOp<blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_resource1blocks/b3/batchnorm/FusedBatchNormV3:batch_mean:04^blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"blocks/b3/batchnorm/AssignNewValue²
$blocks/b3/batchnorm/AssignNewValue_1AssignVariableOp>blocks_b3_batchnorm_fusedbatchnormv3_readvariableop_1_resource5blocks/b3/batchnorm/FusedBatchNormV3:batch_variance:06^blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$blocks/b3/batchnorm/AssignNewValue_1
blocks/b3/prelu/ReluRelu(blocks/b3/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
blocks/b3/prelu/Relu¬
blocks/b3/prelu/ReadVariableOpReadVariableOp'blocks_b3_prelu_readvariableop_resource*"
_output_shapes
:*
dtype02 
blocks/b3/prelu/ReadVariableOp
blocks/b3/prelu/NegNeg&blocks/b3/prelu/ReadVariableOp:value:0*
T0*"
_output_shapes
:2
blocks/b3/prelu/Neg
blocks/b3/prelu/Neg_1Neg(blocks/b3/batchnorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
blocks/b3/prelu/Neg_1
blocks/b3/prelu/Relu_1Relublocks/b3/prelu/Neg_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
blocks/b3/prelu/Relu_1ª
blocks/b3/prelu/mulMulblocks/b3/prelu/Neg:y:0$blocks/b3/prelu/Relu_1:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
blocks/b3/prelu/mulª
blocks/b3/prelu/addAddV2"blocks/b3/prelu/Relu:activations:0blocks/b3/prelu/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
blocks/b3/prelu/addÆ
blocks/b3/maxpool/MaxPoolMaxPoolblocks/b3/prelu/add:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2
blocks/b3/maxpool/MaxPoolô
IdentityIdentity"blocks/b3/maxpool/MaxPool:output:0#^blocks/b3/batchnorm/AssignNewValue%^blocks/b3/batchnorm/AssignNewValue_14^blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp6^blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_1#^blocks/b3/batchnorm/ReadVariableOp%^blocks/b3/batchnorm/ReadVariableOp_1&^blocks/b3/conv/BiasAdd/ReadVariableOp%^blocks/b3/conv/Conv2D/ReadVariableOp^blocks/b3/prelu/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : 2H
"blocks/b3/batchnorm/AssignNewValue"blocks/b3/batchnorm/AssignNewValue2L
$blocks/b3/batchnorm/AssignNewValue_1$blocks/b3/batchnorm/AssignNewValue_12j
3blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp3blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp2n
5blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_15blocks/b3/batchnorm/FusedBatchNormV3/ReadVariableOp_12H
"blocks/b3/batchnorm/ReadVariableOp"blocks/b3/batchnorm/ReadVariableOp2L
$blocks/b3/batchnorm/ReadVariableOp_1$blocks/b3/batchnorm/ReadVariableOp_12N
%blocks/b3/conv/BiasAdd/ReadVariableOp%blocks/b3/conv/BiasAdd/ReadVariableOp2L
$blocks/b3/conv/Conv2D/ReadVariableOp$blocks/b3/conv/Conv2D/ReadVariableOp2@
blocks/b3/prelu/ReadVariableOpblocks/b3/prelu/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


D__inference_blocks/b1_layer_call_and_return_conditional_losses_48931
blocks_b1_conv_input.
blocks_b1_conv_48912:"
blocks_b1_conv_48914:'
blocks_b1_batchnorm_48917:'
blocks_b1_batchnorm_48919:'
blocks_b1_batchnorm_48921:'
blocks_b1_batchnorm_48923:+
blocks_b1_prelu_48926:`
identity¢+blocks/b1/batchnorm/StatefulPartitionedCall¢&blocks/b1/conv/StatefulPartitionedCall¢'blocks/b1/prelu/StatefulPartitionedCallÈ
&blocks/b1/conv/StatefulPartitionedCallStatefulPartitionedCallblocks_b1_conv_inputblocks_b1_conv_48912blocks_b1_conv_48914*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_blocks/b1/conv_layer_call_and_return_conditional_losses_487212(
&blocks/b1/conv/StatefulPartitionedCall¶
+blocks/b1/batchnorm/StatefulPartitionedCallStatefulPartitionedCall/blocks/b1/conv/StatefulPartitionedCall:output:0blocks_b1_batchnorm_48917blocks_b1_batchnorm_48919blocks_b1_batchnorm_48921blocks_b1_batchnorm_48923*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_blocks/b1/batchnorm_layer_call_and_return_conditional_losses_487442-
+blocks/b1/batchnorm/StatefulPartitionedCallÔ
'blocks/b1/prelu/StatefulPartitionedCallStatefulPartitionedCall4blocks/b1/batchnorm/StatefulPartitionedCall:output:0blocks_b1_prelu_48926*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_blocks/b1/prelu_layer_call_and_return_conditional_losses_486842)
'blocks/b1/prelu/StatefulPartitionedCall£
!blocks/b1/maxpool/PartitionedCallPartitionedCall0blocks/b1/prelu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_blocks/b1/maxpool_layer_call_and_return_conditional_losses_486982#
!blocks/b1/maxpool/PartitionedCall
IdentityIdentity*blocks/b1/maxpool/PartitionedCall:output:0,^blocks/b1/batchnorm/StatefulPartitionedCall'^blocks/b1/conv/StatefulPartitionedCall(^blocks/b1/prelu/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ``: : : : : : : 2Z
+blocks/b1/batchnorm/StatefulPartitionedCall+blocks/b1/batchnorm/StatefulPartitionedCall2P
&blocks/b1/conv/StatefulPartitionedCall&blocks/b1/conv/StatefulPartitionedCall2R
'blocks/b1/prelu/StatefulPartitionedCall'blocks/b1/prelu/StatefulPartitionedCall:e a
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
.
_user_specified_nameblocks/b1/conv_input
Á
D
(__inference_restored_function_body_48410

inputs
identity¸
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_insert_channel_axis_layer_call_and_return_conditional_losses_433962
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ``:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
Ð	
õ
D__inference_dense_out_layer_call_and_return_conditional_losses_50461

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


N__inference_blocks/b3/batchnorm_layer_call_and_return_conditional_losses_52648

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È
á
)__inference_simplecnn_layer_call_fn_51176

inputs
unknown:``
	unknown_0:``#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:`#
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13: 

unknown_14:0$

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20: 

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_simplecnn_layer_call_and_return_conditional_losses_504692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs

Î
3__inference_blocks/b1/batchnorm_layer_call_fn_52331

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_blocks/b1/batchnorm_layer_call_and_return_conditional_losses_487442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ``: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
ë)
à
G__inference_fc_batchnorm_layer_call_and_return_conditional_losses_50252

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¤
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decayª
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
£
)__inference_blocks/b2_layer_call_fn_52103

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:0
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_blocks/b2_layer_call_and_return_conditional_losses_492812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ00: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
 
_user_specified_nameinputs


%__inference_dense_layer_call_fn_51833

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_504262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì	
ñ
@__inference_dense_layer_call_and_return_conditional_losses_50426

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
£
.__inference_blocks/b1/conv_layer_call_fn_52282

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_blocks/b1/conv_layer_call_and_return_conditional_losses_487212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ``: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
«
Â
!__inference__traced_restore_52981
file_prefix'
assignvariableop_mean:``-
assignvariableop_1_variance:``"
assignvariableop_2_count:	 1
assignvariableop_3_dense_kernel:+
assignvariableop_4_dense_bias:3
%assignvariableop_5_fc_batchnorm_gamma:2
$assignvariableop_6_fc_batchnorm_beta:9
+assignvariableop_7_fc_batchnorm_moving_mean:=
/assignvariableop_8_fc_batchnorm_moving_variance:/
!assignvariableop_9_fc_prelu_alpha:6
$assignvariableop_10_dense_out_kernel:0
"assignvariableop_11_dense_out_bias:C
)assignvariableop_12_blocks_b1_conv_kernel:5
'assignvariableop_13_blocks_b1_conv_bias:;
-assignvariableop_14_blocks_b1_batchnorm_gamma::
,assignvariableop_15_blocks_b1_batchnorm_beta:A
3assignvariableop_16_blocks_b1_batchnorm_moving_mean:E
7assignvariableop_17_blocks_b1_batchnorm_moving_variance:?
)assignvariableop_18_blocks_b1_prelu_alpha:`C
)assignvariableop_19_blocks_b2_conv_kernel:5
'assignvariableop_20_blocks_b2_conv_bias:;
-assignvariableop_21_blocks_b2_batchnorm_gamma::
,assignvariableop_22_blocks_b2_batchnorm_beta:A
3assignvariableop_23_blocks_b2_batchnorm_moving_mean:E
7assignvariableop_24_blocks_b2_batchnorm_moving_variance:?
)assignvariableop_25_blocks_b2_prelu_alpha:0C
)assignvariableop_26_blocks_b3_conv_kernel:5
'assignvariableop_27_blocks_b3_conv_bias:;
-assignvariableop_28_blocks_b3_batchnorm_gamma::
,assignvariableop_29_blocks_b3_batchnorm_beta:A
3assignvariableop_30_blocks_b3_batchnorm_moving_mean:E
7assignvariableop_31_blocks_b3_batchnorm_moving_variance:?
)assignvariableop_32_blocks_b3_prelu_alpha:#
assignvariableop_33_total: %
assignvariableop_34_count_1: %
assignvariableop_35_total_1: %
assignvariableop_36_count_2: 0
"assignvariableop_37_true_positives:1
#assignvariableop_38_false_negatives:2
$assignvariableop_39_true_positives_1:1
#assignvariableop_40_false_positives:
identity_42¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ø
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*
valueúB÷*B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/alpha/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesâ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¾
_output_shapes«
¨::::::::::::::::::::::::::::::::::::::::::*8
dtypes.
,2*	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_meanIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1 
AssignVariableOp_1AssignVariableOpassignvariableop_1_varianceIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2
AssignVariableOp_2AssignVariableOpassignvariableop_2_countIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¤
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¢
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5ª
AssignVariableOp_5AssignVariableOp%assignvariableop_5_fc_batchnorm_gammaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6©
AssignVariableOp_6AssignVariableOp$assignvariableop_6_fc_batchnorm_betaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7°
AssignVariableOp_7AssignVariableOp+assignvariableop_7_fc_batchnorm_moving_meanIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8´
AssignVariableOp_8AssignVariableOp/assignvariableop_8_fc_batchnorm_moving_varianceIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¦
AssignVariableOp_9AssignVariableOp!assignvariableop_9_fc_prelu_alphaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¬
AssignVariableOp_10AssignVariableOp$assignvariableop_10_dense_out_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ª
AssignVariableOp_11AssignVariableOp"assignvariableop_11_dense_out_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12±
AssignVariableOp_12AssignVariableOp)assignvariableop_12_blocks_b1_conv_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¯
AssignVariableOp_13AssignVariableOp'assignvariableop_13_blocks_b1_conv_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14µ
AssignVariableOp_14AssignVariableOp-assignvariableop_14_blocks_b1_batchnorm_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15´
AssignVariableOp_15AssignVariableOp,assignvariableop_15_blocks_b1_batchnorm_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16»
AssignVariableOp_16AssignVariableOp3assignvariableop_16_blocks_b1_batchnorm_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¿
AssignVariableOp_17AssignVariableOp7assignvariableop_17_blocks_b1_batchnorm_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18±
AssignVariableOp_18AssignVariableOp)assignvariableop_18_blocks_b1_prelu_alphaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19±
AssignVariableOp_19AssignVariableOp)assignvariableop_19_blocks_b2_conv_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¯
AssignVariableOp_20AssignVariableOp'assignvariableop_20_blocks_b2_conv_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21µ
AssignVariableOp_21AssignVariableOp-assignvariableop_21_blocks_b2_batchnorm_gammaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22´
AssignVariableOp_22AssignVariableOp,assignvariableop_22_blocks_b2_batchnorm_betaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23»
AssignVariableOp_23AssignVariableOp3assignvariableop_23_blocks_b2_batchnorm_moving_meanIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¿
AssignVariableOp_24AssignVariableOp7assignvariableop_24_blocks_b2_batchnorm_moving_varianceIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25±
AssignVariableOp_25AssignVariableOp)assignvariableop_25_blocks_b2_prelu_alphaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26±
AssignVariableOp_26AssignVariableOp)assignvariableop_26_blocks_b3_conv_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¯
AssignVariableOp_27AssignVariableOp'assignvariableop_27_blocks_b3_conv_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28µ
AssignVariableOp_28AssignVariableOp-assignvariableop_28_blocks_b3_batchnorm_gammaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29´
AssignVariableOp_29AssignVariableOp,assignvariableop_29_blocks_b3_batchnorm_betaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30»
AssignVariableOp_30AssignVariableOp3assignvariableop_30_blocks_b3_batchnorm_moving_meanIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31¿
AssignVariableOp_31AssignVariableOp7assignvariableop_31_blocks_b3_batchnorm_moving_varianceIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32±
AssignVariableOp_32AssignVariableOp)assignvariableop_32_blocks_b3_prelu_alphaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33¡
AssignVariableOp_33AssignVariableOpassignvariableop_33_totalIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34£
AssignVariableOp_34AssignVariableOpassignvariableop_34_count_1Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35£
AssignVariableOp_35AssignVariableOpassignvariableop_35_total_1Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36£
AssignVariableOp_36AssignVariableOpassignvariableop_36_count_2Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37ª
AssignVariableOp_37AssignVariableOp"assignvariableop_37_true_positivesIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38«
AssignVariableOp_38AssignVariableOp#assignvariableop_38_false_negativesIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39¬
AssignVariableOp_39AssignVariableOp$assignvariableop_39_true_positives_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40«
AssignVariableOp_40AssignVariableOp#assignvariableop_40_false_positivesIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_409
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpä
Identity_41Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_41×
Identity_42IdentityIdentity_41:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_42"#
identity_42Identity_42:output:0*g
_input_shapesV
T: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402(
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
µ


I__inference_blocks/b2/conv_layer_call_and_return_conditional_losses_52435

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ002

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ00: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
 
_user_specified_nameinputs

j
N__inference_insert_channel_axis_layer_call_and_return_conditional_losses_43396

inputs
identityk
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2

ExpandDimso
IdentityIdentityExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ``:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
Î
ã
)__inference_simplecnn_layer_call_fn_50536
input_sp
unknown:``
	unknown_0:``#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:`#
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13: 

unknown_14:0$

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20: 

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_spunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_simplecnn_layer_call_and_return_conditional_losses_504692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
"
_user_specified_name
input_sp
ª

A__inference_blocks_layer_call_and_return_conditional_losses_49965

inputs)
blocks_b1_49919:
blocks_b1_49921:
blocks_b1_49923:
blocks_b1_49925:
blocks_b1_49927:
blocks_b1_49929:%
blocks_b1_49931:`)
blocks_b2_49934:
blocks_b2_49936:
blocks_b2_49938:
blocks_b2_49940:
blocks_b2_49942:
blocks_b2_49944:%
blocks_b2_49946:0)
blocks_b3_49949:
blocks_b3_49951:
blocks_b3_49953:
blocks_b3_49955:
blocks_b3_49957:
blocks_b3_49959:%
blocks_b3_49961:
identity¢!blocks/b1/StatefulPartitionedCall¢!blocks/b2/StatefulPartitionedCall¢!blocks/b3/StatefulPartitionedCallþ
!blocks/b1/StatefulPartitionedCallStatefulPartitionedCallinputsblocks_b1_49919blocks_b1_49921blocks_b1_49923blocks_b1_49925blocks_b1_49927blocks_b1_49929blocks_b1_49931*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_blocks/b1_layer_call_and_return_conditional_losses_488732#
!blocks/b1/StatefulPartitionedCall¢
!blocks/b2/StatefulPartitionedCallStatefulPartitionedCall*blocks/b1/StatefulPartitionedCall:output:0blocks_b2_49934blocks_b2_49936blocks_b2_49938blocks_b2_49940blocks_b2_49942blocks_b2_49944blocks_b2_49946*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_blocks/b2_layer_call_and_return_conditional_losses_492812#
!blocks/b2/StatefulPartitionedCall¢
!blocks/b3/StatefulPartitionedCallStatefulPartitionedCall*blocks/b2/StatefulPartitionedCall:output:0blocks_b3_49949blocks_b3_49951blocks_b3_49953blocks_b3_49955blocks_b3_49957blocks_b3_49959blocks_b3_49961*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_blocks/b3_layer_call_and_return_conditional_losses_496892#
!blocks/b3/StatefulPartitionedCallò
IdentityIdentity*blocks/b3/StatefulPartitionedCall:output:0"^blocks/b1/StatefulPartitionedCall"^blocks/b2/StatefulPartitionedCall"^blocks/b3/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : 2F
!blocks/b1/StatefulPartitionedCall!blocks/b1/StatefulPartitionedCall2F
!blocks/b2/StatefulPartitionedCall!blocks/b2/StatefulPartitionedCall2F
!blocks/b3/StatefulPartitionedCall!blocks/b3/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs

I
-__inference_globalavgpool_layer_call_fn_50168

inputs
identityÒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_globalavgpool_layer_call_and_return_conditional_losses_501622
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*´
serving_default 
A
input_sp5
serving_default_input_sp:0ÿÿÿÿÿÿÿÿÿ``?
probability0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:
à®
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
	optimizer
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api

signatures
ß_default_save_signature
à__call__
+á&call_and_return_all_conditional_losses"Àª
_tf_keras_network£ª{"name": "simplecnn", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "simplecnn", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 96]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_sp"}, "name": "input_sp", "inbound_nodes": []}, {"class_name": "Normalization", "config": {"name": "featurewise_std", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [1, 2]}}, "name": "featurewise_std", "inbound_nodes": [[["input_sp", 0, 0, {}]]]}, {"class_name": "InsertChannelAxis_", "config": {"name": "insert_channel_axis", "trainable": true, "dtype": "float32", "ch_axis": -1}, "name": "insert_channel_axis", "inbound_nodes": [[["featurewise_std", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "blocks", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 96, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "blocks/input"}}, {"class_name": "Sequential", "config": {"name": "blocks/b1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 96, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "blocks/b1/conv_input"}}, {"class_name": "Conv2D", "config": {"name": "blocks/b1/conv", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "blocks/b1/batchnorm", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "PReLU", "config": {"name": "blocks/b1/prelu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [2]}}, {"class_name": "MaxPooling2D", "config": {"name": "blocks/b1/maxpool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}}, {"class_name": "Sequential", "config": {"name": "blocks/b2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "blocks/b2/conv_input"}}, {"class_name": "Conv2D", "config": {"name": "blocks/b2/conv", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "blocks/b2/batchnorm", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "PReLU", "config": {"name": "blocks/b2/prelu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [2]}}, {"class_name": "MaxPooling2D", "config": {"name": "blocks/b2/maxpool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}}, {"class_name": "Sequential", "config": {"name": "blocks/b3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 24, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "blocks/b3/conv_input"}}, {"class_name": "Conv2D", "config": {"name": "blocks/b3/conv", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "blocks/b3/batchnorm", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "PReLU", "config": {"name": "blocks/b3/prelu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [2]}}, {"class_name": "MaxPooling2D", "config": {"name": "blocks/b3/maxpool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}}]}, "name": "blocks", "inbound_nodes": [[["insert_channel_axis", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "globalavgpool", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "globalavgpool", "inbound_nodes": [[["blocks", 1, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["globalavgpool", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "fc_batchnorm", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "fc_batchnorm", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "fc_dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "fc_dropout", "inbound_nodes": [[["fc_batchnorm", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "fc_prelu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1]}, "name": "fc_prelu", "inbound_nodes": [[["fc_dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_out", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_out", "inbound_nodes": [[["fc_prelu", 0, 0, {}]]]}, {"class_name": "Sigmoid_", "config": {"name": "probability", "trainable": true, "dtype": "float32"}, "name": "probability", "inbound_nodes": [[["dense_out", 0, 0, {}]]]}], "input_layers": [["input_sp", 0, 0]], "output_layers": [["probability", 0, 0]]}, "shared_object_id": 60, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 96, 96]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 96]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 96, 96]}, "float32", "input_sp"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "simplecnn", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 96]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_sp"}, "name": "input_sp", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Normalization", "config": {"name": "featurewise_std", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [1, 2]}}, "name": "featurewise_std", "inbound_nodes": [[["input_sp", 0, 0, {}]]], "shared_object_id": 1}, {"class_name": "InsertChannelAxis_", "config": {"name": "insert_channel_axis", "trainable": true, "dtype": "float32", "ch_axis": -1}, "name": "insert_channel_axis", "inbound_nodes": [[["featurewise_std", 0, 0, {}]]], "shared_object_id": 2}, {"class_name": "Sequential", "config": {"name": "blocks", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 96, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "blocks/input"}}, {"class_name": "Sequential", "config": {"name": "blocks/b1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 96, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "blocks/b1/conv_input"}}, {"class_name": "Conv2D", "config": {"name": "blocks/b1/conv", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "blocks/b1/batchnorm", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "PReLU", "config": {"name": "blocks/b1/prelu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [2]}}, {"class_name": "MaxPooling2D", "config": {"name": "blocks/b1/maxpool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}}, {"class_name": "Sequential", "config": {"name": "blocks/b2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "blocks/b2/conv_input"}}, {"class_name": "Conv2D", "config": {"name": "blocks/b2/conv", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "blocks/b2/batchnorm", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "PReLU", "config": {"name": "blocks/b2/prelu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [2]}}, {"class_name": "MaxPooling2D", "config": {"name": "blocks/b2/maxpool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}}, {"class_name": "Sequential", "config": {"name": "blocks/b3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 24, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "blocks/b3/conv_input"}}, {"class_name": "Conv2D", "config": {"name": "blocks/b3/conv", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "blocks/b3/batchnorm", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "PReLU", "config": {"name": "blocks/b3/prelu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [2]}}, {"class_name": "MaxPooling2D", "config": {"name": "blocks/b3/maxpool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}}]}, "name": "blocks", "inbound_nodes": [[["insert_channel_axis", 0, 0, {}]]], "shared_object_id": 43}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "globalavgpool", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "globalavgpool", "inbound_nodes": [[["blocks", 1, 0, {}]]], "shared_object_id": 44}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 45}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 46}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["globalavgpool", 0, 0, {}]]], "shared_object_id": 47}, {"class_name": "BatchNormalization", "config": {"name": "fc_batchnorm", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 48}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 49}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 50}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 51}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "fc_batchnorm", "inbound_nodes": [[["dense", 0, 0, {}]]], "shared_object_id": 52}, {"class_name": "Dropout", "config": {"name": "fc_dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "fc_dropout", "inbound_nodes": [[["fc_batchnorm", 0, 0, {}]]], "shared_object_id": 53}, {"class_name": "PReLU", "config": {"name": "fc_prelu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 54}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1]}, "name": "fc_prelu", "inbound_nodes": [[["fc_dropout", 0, 0, {}]]], "shared_object_id": 55}, {"class_name": "Dense", "config": {"name": "dense_out", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 56}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 57}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_out", "inbound_nodes": [[["fc_prelu", 0, 0, {}]]], "shared_object_id": 58}, {"class_name": "Sigmoid_", "config": {"name": "probability", "trainable": true, "dtype": "float32"}, "name": "probability", "inbound_nodes": [[["dense_out", 0, 0, {}]]], "shared_object_id": 59}], "input_layers": [["input_sp", 0, 0]], "output_layers": [["probability", 0, 0]]}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "sum_over_batch_size", "name": "bce_loss", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 62}, "metrics": [[{"class_name": "BinaryAccuracy", "config": {"name": "accuracy", "dtype": "float32", "threshold": 0.5}, "shared_object_id": 63}, {"class_name": "Recall", "config": {"name": "recall", "dtype": "float32", "thresholds": 0.5, "top_k": null, "class_id": null}, "shared_object_id": 64}, {"class_name": "Precision", "config": {"name": "precision", "dtype": "float32", "thresholds": 0.5, "top_k": null, "class_id": null}, "shared_object_id": 65}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 5e-05, "decay": 0.004, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07}}}}

#_self_saveable_object_factories"ò
_tf_keras_input_layerÒ{"class_name": "InputLayer", "name": "input_sp", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 96]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 96]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_sp"}}


_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean
variance
	count
state_variables
#_self_saveable_object_factories
	keras_api"Ù
_tf_keras_layer¿{"name": "featurewise_std", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null]}, "stateful": false, "must_restore_from_config": true, "class_name": "Normalization", "config": {"name": "featurewise_std", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [1, 2]}}, "inbound_nodes": [[["input_sp", 0, 0, {}]]], "shared_object_id": 1, "build_input_shape": [512, 96, 96]}
Ö
#_self_saveable_object_factories
	variables
 regularization_losses
!trainable_variables
"	keras_api
â__call__
+ã&call_and_return_all_conditional_losses" 
_tf_keras_layer{"name": "insert_channel_axis", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "InsertChannelAxis_", "config": {"name": "insert_channel_axis", "trainable": true, "dtype": "float32", "ch_axis": -1}, "inbound_nodes": [[["featurewise_std", 0, 0, {}]]], "shared_object_id": 2}
ál
#layer_with_weights-0
#layer-0
$layer_with_weights-1
$layer-1
%layer_with_weights-2
%layer-2
#&_self_saveable_object_factories
'	variables
(regularization_losses
)trainable_variables
*	keras_api
ä__call__
+å&call_and_return_all_conditional_losses"¶j
_tf_keras_sequentialj{"name": "blocks", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "blocks", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 96, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "blocks/input"}}, {"class_name": "Sequential", "config": {"name": "blocks/b1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 96, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "blocks/b1/conv_input"}}, {"class_name": "Conv2D", "config": {"name": "blocks/b1/conv", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "blocks/b1/batchnorm", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "PReLU", "config": {"name": "blocks/b1/prelu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [2]}}, {"class_name": "MaxPooling2D", "config": {"name": "blocks/b1/maxpool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}}, {"class_name": "Sequential", "config": {"name": "blocks/b2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "blocks/b2/conv_input"}}, {"class_name": "Conv2D", "config": {"name": "blocks/b2/conv", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "blocks/b2/batchnorm", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "PReLU", "config": {"name": "blocks/b2/prelu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [2]}}, {"class_name": "MaxPooling2D", "config": {"name": "blocks/b2/maxpool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}}, {"class_name": "Sequential", "config": {"name": "blocks/b3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 24, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "blocks/b3/conv_input"}}, {"class_name": "Conv2D", "config": {"name": "blocks/b3/conv", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "blocks/b3/batchnorm", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "PReLU", "config": {"name": "blocks/b3/prelu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [2]}}, {"class_name": "MaxPooling2D", "config": {"name": "blocks/b3/maxpool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}}]}, "inbound_nodes": [[["insert_channel_axis", 0, 0, {}]]], "shared_object_id": 43, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 66}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 96, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 96, 96, 1]}, "float32", "blocks/input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "blocks", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 96, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "blocks/input"}, "shared_object_id": 3}, {"class_name": "Sequential", "config": {"name": "blocks/b1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 96, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "blocks/b1/conv_input"}}, {"class_name": "Conv2D", "config": {"name": "blocks/b1/conv", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "blocks/b1/batchnorm", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "PReLU", "config": {"name": "blocks/b1/prelu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [2]}}, {"class_name": "MaxPooling2D", "config": {"name": "blocks/b1/maxpool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "shared_object_id": 16}, {"class_name": "Sequential", "config": {"name": "blocks/b2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "blocks/b2/conv_input"}}, {"class_name": "Conv2D", "config": {"name": "blocks/b2/conv", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "blocks/b2/batchnorm", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "PReLU", "config": {"name": "blocks/b2/prelu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [2]}}, {"class_name": "MaxPooling2D", "config": {"name": "blocks/b2/maxpool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "shared_object_id": 29}, {"class_name": "Sequential", "config": {"name": "blocks/b3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 24, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "blocks/b3/conv_input"}}, {"class_name": "Conv2D", "config": {"name": "blocks/b3/conv", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "blocks/b3/batchnorm", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "PReLU", "config": {"name": "blocks/b3/prelu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [2]}}, {"class_name": "MaxPooling2D", "config": {"name": "blocks/b3/maxpool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "shared_object_id": 42}]}}}
ÿ
#+_self_saveable_object_factories
,	variables
-regularization_losses
.trainable_variables
/	keras_api
æ__call__
+ç&call_and_return_all_conditional_losses"É
_tf_keras_layer¯{"name": "globalavgpool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GlobalAveragePooling2D", "config": {"name": "globalavgpool", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["blocks", 1, 0, {}]]], "shared_object_id": 44, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 67}}
§	

0kernel
1bias
#2_self_saveable_object_factories
3	variables
4regularization_losses
5trainable_variables
6	keras_api
è__call__
+é&call_and_return_all_conditional_losses"Û
_tf_keras_layerÁ{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 45}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 46}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["globalavgpool", 0, 0, {}]]], "shared_object_id": 47, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}, "shared_object_id": 68}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}

7axis
	8gamma
9beta
:moving_mean
;moving_variance
#<_self_saveable_object_factories
=	variables
>regularization_losses
?trainable_variables
@	keras_api
ê__call__
+ë&call_and_return_all_conditional_losses"	
_tf_keras_layerì{"name": "fc_batchnorm", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "fc_batchnorm", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 48}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 49}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 50}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 51}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["dense", 0, 0, {}]]], "shared_object_id": 52, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 16}}, "shared_object_id": 69}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
×
#A_self_saveable_object_factories
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
ì__call__
+í&call_and_return_all_conditional_losses"¡
_tf_keras_layer{"name": "fc_dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "fc_dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "inbound_nodes": [[["fc_batchnorm", 0, 0, {}]]], "shared_object_id": 53}
Ê
Fshared_axes
	Galpha
#H_self_saveable_object_factories
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
î__call__
+ï&call_and_return_all_conditional_losses"ø
_tf_keras_layerÞ{"name": "fc_prelu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "PReLU", "config": {"name": "fc_prelu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 54}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [1]}, "inbound_nodes": [[["fc_dropout", 0, 0, {}]]], "shared_object_id": 55, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 70}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
©	

Mkernel
Nbias
#O_self_saveable_object_factories
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
ð__call__
+ñ&call_and_return_all_conditional_losses"Ý
_tf_keras_layerÃ{"name": "dense_out", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_out", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 56}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 57}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["fc_prelu", 0, 0, {}]]], "shared_object_id": 58, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}, "shared_object_id": 71}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
¨
#T_self_saveable_object_factories
U	variables
Vregularization_losses
Wtrainable_variables
X	keras_api
ò__call__
+ó&call_and_return_all_conditional_losses"ò
_tf_keras_layerØ{"name": "probability", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Sigmoid_", "config": {"name": "probability", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["dense_out", 0, 0, {}]]], "shared_object_id": 59}
"
	optimizer
 "
trackable_dict_wrapper

0
1
2
Y3
Z4
[5
\6
]7
^8
_9
`10
a11
b12
c13
d14
e15
f16
g17
h18
i19
j20
k21
l22
m23
024
125
826
927
:28
;29
G30
M31
N32"
trackable_list_wrapper
 "
trackable_list_wrapper
Æ
Y0
Z1
[2
\3
_4
`5
a6
b7
c8
f9
g10
h11
i12
j13
m14
015
116
817
918
G19
M20
N21"
trackable_list_wrapper
Î
	variables
regularization_losses
nmetrics
trainable_variables
olayer_metrics

players
qnon_trainable_variables
rlayer_regularization_losses
à__call__
ß_default_save_signature
+á&call_and_return_all_conditional_losses
'á"call_and_return_conditional_losses"
_generic_user_object
-
ôserving_default"
signature_map
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
:``2mean
:``2variance
:	 2count
C
mean
variance
	count"
trackable_dict_wrapper
 "
trackable_dict_wrapper
"
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
	variables
 regularization_losses
smetrics
!trainable_variables
tlayer_metrics

ulayers
vnon_trainable_variables
wlayer_regularization_losses
â__call__
+ã&call_and_return_all_conditional_losses
'ã"call_and_return_conditional_losses"
_generic_user_object
*
xlayer_with_weights-0
xlayer-0
ylayer_with_weights-1
ylayer-1
zlayer_with_weights-2
zlayer-2
{layer-3
#|_self_saveable_object_factories
}	variables
~regularization_losses
trainable_variables
	keras_api
õ__call__
+ö&call_and_return_all_conditional_losses"Ü'
_tf_keras_sequential½'{"name": "blocks/b1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "blocks/b1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 96, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "blocks/b1/conv_input"}}, {"class_name": "Conv2D", "config": {"name": "blocks/b1/conv", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "blocks/b1/batchnorm", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "PReLU", "config": {"name": "blocks/b1/prelu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [2]}}, {"class_name": "MaxPooling2D", "config": {"name": "blocks/b1/maxpool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 66}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 96, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 96, 96, 1]}, "float32", "blocks/b1/conv_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "blocks/b1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 96, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "blocks/b1/conv_input"}, "shared_object_id": 4}, {"class_name": "Conv2D", "config": {"name": "blocks/b1/conv", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 7}, {"class_name": "BatchNormalization", "config": {"name": "blocks/b1/batchnorm", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 9}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 11}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 12}, {"class_name": "PReLU", "config": {"name": "blocks/b1/prelu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [2]}, "shared_object_id": 14}, {"class_name": "MaxPooling2D", "config": {"name": "blocks/b1/maxpool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 15}]}}}
«*
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
$_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
÷__call__
+ø&call_and_return_all_conditional_losses"ç'
_tf_keras_sequentialÈ'{"name": "blocks/b2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "blocks/b2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "blocks/b2/conv_input"}}, {"class_name": "Conv2D", "config": {"name": "blocks/b2/conv", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "blocks/b2/batchnorm", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "PReLU", "config": {"name": "blocks/b2/prelu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [2]}}, {"class_name": "MaxPooling2D", "config": {"name": "blocks/b2/maxpool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "shared_object_id": 29, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}, "shared_object_id": 72}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 16]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 48, 48, 16]}, "float32", "blocks/b2/conv_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "blocks/b2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "blocks/b2/conv_input"}, "shared_object_id": 17}, {"class_name": "Conv2D", "config": {"name": "blocks/b2/conv", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20}, {"class_name": "BatchNormalization", "config": {"name": "blocks/b2/batchnorm", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 22}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 24}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 25}, {"class_name": "PReLU", "config": {"name": "blocks/b2/prelu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [2]}, "shared_object_id": 27}, {"class_name": "MaxPooling2D", "config": {"name": "blocks/b2/maxpool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 28}]}}}
«*
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
$_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
ù__call__
+ú&call_and_return_all_conditional_losses"ç'
_tf_keras_sequentialÈ'{"name": "blocks/b3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "blocks/b3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 24, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "blocks/b3/conv_input"}}, {"class_name": "Conv2D", "config": {"name": "blocks/b3/conv", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "blocks/b3/batchnorm", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "PReLU", "config": {"name": "blocks/b3/prelu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [2]}}, {"class_name": "MaxPooling2D", "config": {"name": "blocks/b3/maxpool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "shared_object_id": 42, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}, "shared_object_id": 73}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 16]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 24, 24, 16]}, "float32", "blocks/b3/conv_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "blocks/b3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 24, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "blocks/b3/conv_input"}, "shared_object_id": 30}, {"class_name": "Conv2D", "config": {"name": "blocks/b3/conv", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 33}, {"class_name": "BatchNormalization", "config": {"name": "blocks/b3/batchnorm", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 34}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 35}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 37}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 38}, {"class_name": "PReLU", "config": {"name": "blocks/b3/prelu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 39}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [2]}, "shared_object_id": 40}, {"class_name": "MaxPooling2D", "config": {"name": "blocks/b3/maxpool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 41}]}}}
 "
trackable_dict_wrapper
¾
Y0
Z1
[2
\3
]4
^5
_6
`7
a8
b9
c10
d11
e12
f13
g14
h15
i16
j17
k18
l19
m20"
trackable_list_wrapper
 "
trackable_list_wrapper

Y0
Z1
[2
\3
_4
`5
a6
b7
c8
f9
g10
h11
i12
j13
m14"
trackable_list_wrapper
µ
'	variables
(regularization_losses
metrics
)trainable_variables
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
ä__call__
+å&call_and_return_all_conditional_losses
'å"call_and_return_conditional_losses"
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
,	variables
-regularization_losses
metrics
.trainable_variables
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
æ__call__
+ç&call_and_return_all_conditional_losses
'ç"call_and_return_conditional_losses"
_generic_user_object
:2dense/kernel
:2
dense/bias
 "
trackable_dict_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
µ
3	variables
4regularization_losses
metrics
5trainable_variables
layer_metrics
layers
 non_trainable_variables
 ¡layer_regularization_losses
è__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 :2fc_batchnorm/gamma
:2fc_batchnorm/beta
(:& (2fc_batchnorm/moving_mean
,:* (2fc_batchnorm/moving_variance
 "
trackable_dict_wrapper
<
80
91
:2
;3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
µ
=	variables
>regularization_losses
¢metrics
?trainable_variables
£layer_metrics
¤layers
¥non_trainable_variables
 ¦layer_regularization_losses
ê__call__
+ë&call_and_return_all_conditional_losses
'ë"call_and_return_conditional_losses"
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
B	variables
Cregularization_losses
§metrics
Dtrainable_variables
¨layer_metrics
©layers
ªnon_trainable_variables
 «layer_regularization_losses
ì__call__
+í&call_and_return_all_conditional_losses
'í"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:2fc_prelu/alpha
 "
trackable_dict_wrapper
'
G0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
G0"
trackable_list_wrapper
µ
I	variables
Jregularization_losses
¬metrics
Ktrainable_variables
­layer_metrics
®layers
¯non_trainable_variables
 °layer_regularization_losses
î__call__
+ï&call_and_return_all_conditional_losses
'ï"call_and_return_conditional_losses"
_generic_user_object
": 2dense_out/kernel
:2dense_out/bias
 "
trackable_dict_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
µ
P	variables
Qregularization_losses
±metrics
Rtrainable_variables
²layer_metrics
³layers
´non_trainable_variables
 µlayer_regularization_losses
ð__call__
+ñ&call_and_return_all_conditional_losses
'ñ"call_and_return_conditional_losses"
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
U	variables
Vregularization_losses
¶metrics
Wtrainable_variables
·layer_metrics
¸layers
¹non_trainable_variables
 ºlayer_regularization_losses
ò__call__
+ó&call_and_return_all_conditional_losses
'ó"call_and_return_conditional_losses"
_generic_user_object
/:-2blocks/b1/conv/kernel
!:2blocks/b1/conv/bias
':%2blocks/b1/batchnorm/gamma
&:$2blocks/b1/batchnorm/beta
/:- (2blocks/b1/batchnorm/moving_mean
3:1 (2#blocks/b1/batchnorm/moving_variance
+:)`2blocks/b1/prelu/alpha
/:-2blocks/b2/conv/kernel
!:2blocks/b2/conv/bias
':%2blocks/b2/batchnorm/gamma
&:$2blocks/b2/batchnorm/beta
/:- (2blocks/b2/batchnorm/moving_mean
3:1 (2#blocks/b2/batchnorm/moving_variance
+:)02blocks/b2/prelu/alpha
/:-2blocks/b3/conv/kernel
!:2blocks/b3/conv/bias
':%2blocks/b3/batchnorm/gamma
&:$2blocks/b3/batchnorm/beta
/:- (2blocks/b3/batchnorm/moving_mean
3:1 (2#blocks/b3/batchnorm/moving_variance
+:)2blocks/b3/prelu/alpha
@
»0
¼1
½2
¾3"
trackable_list_wrapper
 "
trackable_dict_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
n
0
1
2
]3
^4
d5
e6
k7
l8
:9
;10"
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


Ykernel
Zbias
$¿_self_saveable_object_factories
À	variables
Áregularization_losses
Âtrainable_variables
Ã	keras_api
û__call__
+ü&call_and_return_all_conditional_losses"¶	
_tf_keras_layer	{"name": "blocks/b1/conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "blocks/b1/conv", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 66}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 96, 1]}}
ñ

	Äaxis
	[gamma
\beta
]moving_mean
^moving_variance
$Å_self_saveable_object_factories
Æ	variables
Çregularization_losses
Ètrainable_variables
É	keras_api
ý__call__
+þ&call_and_return_all_conditional_losses"ð
_tf_keras_layerÖ{"name": "blocks/b1/batchnorm", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "blocks/b1/batchnorm", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 9}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 11}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}, "shared_object_id": 74}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 96, 16]}}
Ç
Êshared_axes
	_alpha
$Ë_self_saveable_object_factories
Ì	variables
Íregularization_losses
Îtrainable_variables
Ï	keras_api
ÿ__call__
+&call_and_return_all_conditional_losses"ï
_tf_keras_layerÕ{"name": "blocks/b1/prelu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "PReLU", "config": {"name": "blocks/b1/prelu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [2]}, "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"1": 96, "3": 16}}, "shared_object_id": 75}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 96, 16]}}
Þ
$Ð_self_saveable_object_factories
Ñ	variables
Òregularization_losses
Ótrainable_variables
Ô	keras_api
__call__
+&call_and_return_all_conditional_losses"£
_tf_keras_layer{"name": "blocks/b1/maxpool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "blocks/b1/maxpool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 76}}
 "
trackable_dict_wrapper
Q
Y0
Z1
[2
\3
]4
^5
_6"
trackable_list_wrapper
 "
trackable_list_wrapper
C
Y0
Z1
[2
\3
_4"
trackable_list_wrapper
µ
}	variables
~regularization_losses
Õmetrics
trainable_variables
Ölayer_metrics
×layers
Ønon_trainable_variables
 Ùlayer_regularization_losses
õ__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses"
_generic_user_object


`kernel
abias
$Ú_self_saveable_object_factories
Û	variables
Üregularization_losses
Ýtrainable_variables
Þ	keras_api
__call__
+&call_and_return_all_conditional_losses"»	
_tf_keras_layer¡	{"name": "blocks/b2/conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "blocks/b2/conv", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}, "shared_object_id": 72}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 16]}}
ó

	ßaxis
	bgamma
cbeta
dmoving_mean
emoving_variance
$à_self_saveable_object_factories
á	variables
âregularization_losses
ãtrainable_variables
ä	keras_api
__call__
+&call_and_return_all_conditional_losses"ò
_tf_keras_layerØ{"name": "blocks/b2/batchnorm", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "blocks/b2/batchnorm", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 22}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 24}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}, "shared_object_id": 77}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 16]}}
Ç
åshared_axes
	falpha
$æ_self_saveable_object_factories
ç	variables
èregularization_losses
étrainable_variables
ê	keras_api
__call__
+&call_and_return_all_conditional_losses"ï
_tf_keras_layerÕ{"name": "blocks/b2/prelu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "PReLU", "config": {"name": "blocks/b2/prelu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [2]}, "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"1": 48, "3": 16}}, "shared_object_id": 78}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 16]}}
Þ
$ë_self_saveable_object_factories
ì	variables
íregularization_losses
îtrainable_variables
ï	keras_api
__call__
+&call_and_return_all_conditional_losses"£
_tf_keras_layer{"name": "blocks/b2/maxpool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "blocks/b2/maxpool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 28, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 79}}
 "
trackable_dict_wrapper
Q
`0
a1
b2
c3
d4
e5
f6"
trackable_list_wrapper
 "
trackable_list_wrapper
C
`0
a1
b2
c3
f4"
trackable_list_wrapper
¸
	variables
regularization_losses
ðmetrics
trainable_variables
ñlayer_metrics
òlayers
ónon_trainable_variables
 ôlayer_regularization_losses
÷__call__
+ø&call_and_return_all_conditional_losses
'ø"call_and_return_conditional_losses"
_generic_user_object


gkernel
hbias
$õ_self_saveable_object_factories
ö	variables
÷regularization_losses
øtrainable_variables
ù	keras_api
__call__
+&call_and_return_all_conditional_losses"»	
_tf_keras_layer¡	{"name": "blocks/b3/conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "blocks/b3/conv", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 33, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}, "shared_object_id": 73}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 16]}}
ó

	úaxis
	igamma
jbeta
kmoving_mean
lmoving_variance
$û_self_saveable_object_factories
ü	variables
ýregularization_losses
þtrainable_variables
ÿ	keras_api
__call__
+&call_and_return_all_conditional_losses"ò
_tf_keras_layerØ{"name": "blocks/b3/batchnorm", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "blocks/b3/batchnorm", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 34}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 35}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 37}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 38, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}, "shared_object_id": 80}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 16]}}
Ç
shared_axes
	malpha
$_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"ï
_tf_keras_layerÕ{"name": "blocks/b3/prelu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "PReLU", "config": {"name": "blocks/b3/prelu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 39}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": [2]}, "shared_object_id": 40, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"1": 24, "3": 16}}, "shared_object_id": 81}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 16]}}
Þ
$_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"£
_tf_keras_layer{"name": "blocks/b3/maxpool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "blocks/b3/maxpool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 41, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 82}}
 "
trackable_dict_wrapper
Q
g0
h1
i2
j3
k4
l5
m6"
trackable_list_wrapper
 "
trackable_list_wrapper
C
g0
h1
i2
j3
m4"
trackable_list_wrapper
¸
	variables
regularization_losses
metrics
trainable_variables
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
ù__call__
+ú&call_and_return_all_conditional_losses
'ú"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
#0
$1
%2"
trackable_list_wrapper
J
]0
^1
d2
e3
k4
l5"
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
.
:0
;1"
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
Ø

total

count
	variables
	keras_api"
_tf_keras_metric{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 83}


total

count

_fn_kwargs
	variables
	keras_api"Á
_tf_keras_metric¦{"class_name": "BinaryAccuracy", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "threshold": 0.5}, "shared_object_id": 63}
¶

thresholds
true_positives
false_negatives
	variables
	keras_api"×
_tf_keras_metric¼{"class_name": "Recall", "name": "recall", "dtype": "float32", "config": {"name": "recall", "dtype": "float32", "thresholds": 0.5, "top_k": null, "class_id": null}, "shared_object_id": 64}
¿

thresholds
true_positives
 false_positives
¡	variables
¢	keras_api"à
_tf_keras_metricÅ{"class_name": "Precision", "name": "precision", "dtype": "float32", "config": {"name": "precision", "dtype": "float32", "thresholds": 0.5, "top_k": null, "class_id": null}, "shared_object_id": 65}
 "
trackable_dict_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
¸
À	variables
Áregularization_losses
£metrics
Âtrainable_variables
¤layer_metrics
¥layers
¦non_trainable_variables
 §layer_regularization_losses
û__call__
+ü&call_and_return_all_conditional_losses
'ü"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
[0
\1
]2
^3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
¸
Æ	variables
Çregularization_losses
¨metrics
Ètrainable_variables
©layer_metrics
ªlayers
«non_trainable_variables
 ¬layer_regularization_losses
ý__call__
+þ&call_and_return_all_conditional_losses
'þ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
_0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
_0"
trackable_list_wrapper
¸
Ì	variables
Íregularization_losses
­metrics
Îtrainable_variables
®layer_metrics
¯layers
°non_trainable_variables
 ±layer_regularization_losses
ÿ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ñ	variables
Òregularization_losses
²metrics
Ótrainable_variables
³layer_metrics
´layers
µnon_trainable_variables
 ¶layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
x0
y1
z2
{3"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
¸
Û	variables
Üregularization_losses
·metrics
Ýtrainable_variables
¸layer_metrics
¹layers
ºnon_trainable_variables
 »layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
b0
c1
d2
e3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
¸
á	variables
âregularization_losses
¼metrics
ãtrainable_variables
½layer_metrics
¾layers
¿non_trainable_variables
 Àlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
f0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
f0"
trackable_list_wrapper
¸
ç	variables
èregularization_losses
Ámetrics
étrainable_variables
Âlayer_metrics
Ãlayers
Änon_trainable_variables
 Ålayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ì	variables
íregularization_losses
Æmetrics
îtrainable_variables
Çlayer_metrics
Èlayers
Énon_trainable_variables
 Êlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
@
0
1
2
3"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
¸
ö	variables
÷regularization_losses
Ëmetrics
øtrainable_variables
Ìlayer_metrics
Ílayers
Înon_trainable_variables
 Ïlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
i0
j1
k2
l3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
¸
ü	variables
ýregularization_losses
Ðmetrics
þtrainable_variables
Ñlayer_metrics
Òlayers
Ónon_trainable_variables
 Ôlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
m0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
m0"
trackable_list_wrapper
¸
	variables
regularization_losses
Õmetrics
trainable_variables
Ölayer_metrics
×layers
Ønon_trainable_variables
 Ùlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
regularization_losses
Úmetrics
trainable_variables
Ûlayer_metrics
Ülayers
Ýnon_trainable_variables
 Þlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
@
0
1
2
3"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
0
 1"
trackable_list_wrapper
.
¡	variables"
_generic_user_object
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
.
]0
^1"
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
.
d0
e1"
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
.
k0
l1"
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
ã2à
 __inference__wrapped_model_48545»
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
annotationsª *+¢(
&#
input_spÿÿÿÿÿÿÿÿÿ``
ò2ï
)__inference_simplecnn_layer_call_fn_50536
)__inference_simplecnn_layer_call_fn_51176
)__inference_simplecnn_layer_call_fn_51245
)__inference_simplecnn_layer_call_fn_50864À
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
kwonlydefaultsª 
annotationsª *
 
Þ2Û
D__inference_simplecnn_layer_call_and_return_conditional_losses_51390
D__inference_simplecnn_layer_call_and_return_conditional_losses_51548
D__inference_simplecnn_layer_call_and_return_conditional_losses_50950
D__inference_simplecnn_layer_call_and_return_conditional_losses_51036À
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
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
3__inference_insert_channel_axis_layer_call_fn_42951
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
annotationsª *
 
î2ë
N__inference_insert_channel_axis_layer_call_and_return_conditional_losses_43396
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
annotationsª *
 
æ2ã
&__inference_blocks_layer_call_fn_49867
&__inference_blocks_layer_call_fn_51595
&__inference_blocks_layer_call_fn_51642
&__inference_blocks_layer_call_fn_50057À
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
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
A__inference_blocks_layer_call_and_return_conditional_losses_51733
A__inference_blocks_layer_call_and_return_conditional_losses_51824
A__inference_blocks_layer_call_and_return_conditional_losses_50106
A__inference_blocks_layer_call_and_return_conditional_losses_50155À
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
kwonlydefaultsª 
annotationsª *
 
2
-__inference_globalavgpool_layer_call_fn_50168à
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
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
°2­
H__inference_globalavgpool_layer_call_and_return_conditional_losses_50162à
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
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ï2Ì
%__inference_dense_layer_call_fn_51833¢
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
annotationsª *
 
ê2ç
@__inference_dense_layer_call_and_return_conditional_losses_51843¢
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
annotationsª *
 
2
,__inference_fc_batchnorm_layer_call_fn_51856
,__inference_fc_batchnorm_layer_call_fn_51869´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ì2É
G__inference_fc_batchnorm_layer_call_and_return_conditional_losses_51889
G__inference_fc_batchnorm_layer_call_and_return_conditional_losses_51923´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
*__inference_fc_dropout_layer_call_fn_51928
*__inference_fc_dropout_layer_call_fn_51933´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_fc_dropout_layer_call_and_return_conditional_losses_51938
E__inference_fc_dropout_layer_call_and_return_conditional_losses_51942´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ö2ó
(__inference_fc_prelu_layer_call_fn_50351Æ
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
annotationsª *&¢#
!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
C__inference_fc_prelu_layer_call_and_return_conditional_losses_50343Æ
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
annotationsª *&¢#
!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ó2Ð
)__inference_dense_out_layer_call_fn_51951¢
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
annotationsª *
 
î2ë
D__inference_dense_out_layer_call_and_return_conditional_losses_51961¢
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
annotationsª *
 
Ë2È
+__inference_probability_layer_call_fn_42553
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
annotationsª *
 
æ2ã
F__inference_probability_layer_call_and_return_conditional_losses_43277
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
annotationsª *
 
ËBÈ
#__inference_signature_wrapper_51107input_sp"
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
annotationsª *
 
ò2ï
)__inference_blocks/b1_layer_call_fn_48776
)__inference_blocks/b1_layer_call_fn_51980
)__inference_blocks/b1_layer_call_fn_51999
)__inference_blocks/b1_layer_call_fn_48909À
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
kwonlydefaultsª 
annotationsª *
 
Þ2Û
D__inference_blocks/b1_layer_call_and_return_conditional_losses_52032
D__inference_blocks/b1_layer_call_and_return_conditional_losses_52065
D__inference_blocks/b1_layer_call_and_return_conditional_losses_48931
D__inference_blocks/b1_layer_call_and_return_conditional_losses_48953À
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
kwonlydefaultsª 
annotationsª *
 
ò2ï
)__inference_blocks/b2_layer_call_fn_49184
)__inference_blocks/b2_layer_call_fn_52084
)__inference_blocks/b2_layer_call_fn_52103
)__inference_blocks/b2_layer_call_fn_49317À
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
kwonlydefaultsª 
annotationsª *
 
Þ2Û
D__inference_blocks/b2_layer_call_and_return_conditional_losses_52136
D__inference_blocks/b2_layer_call_and_return_conditional_losses_52169
D__inference_blocks/b2_layer_call_and_return_conditional_losses_49339
D__inference_blocks/b2_layer_call_and_return_conditional_losses_49361À
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
kwonlydefaultsª 
annotationsª *
 
ò2ï
)__inference_blocks/b3_layer_call_fn_49592
)__inference_blocks/b3_layer_call_fn_52188
)__inference_blocks/b3_layer_call_fn_52207
)__inference_blocks/b3_layer_call_fn_49725À
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
kwonlydefaultsª 
annotationsª *
 
Þ2Û
D__inference_blocks/b3_layer_call_and_return_conditional_losses_52240
D__inference_blocks/b3_layer_call_and_return_conditional_losses_52273
D__inference_blocks/b3_layer_call_and_return_conditional_losses_49747
D__inference_blocks/b3_layer_call_and_return_conditional_losses_49769À
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
kwonlydefaultsª 
annotationsª *
 
Ø2Õ
.__inference_blocks/b1/conv_layer_call_fn_52282¢
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
annotationsª *
 
ó2ð
I__inference_blocks/b1/conv_layer_call_and_return_conditional_losses_52292¢
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
annotationsª *
 
2
3__inference_blocks/b1/batchnorm_layer_call_fn_52305
3__inference_blocks/b1/batchnorm_layer_call_fn_52318
3__inference_blocks/b1/batchnorm_layer_call_fn_52331
3__inference_blocks/b1/batchnorm_layer_call_fn_52344´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ú2÷
N__inference_blocks/b1/batchnorm_layer_call_and_return_conditional_losses_52362
N__inference_blocks/b1/batchnorm_layer_call_and_return_conditional_losses_52380
N__inference_blocks/b1/batchnorm_layer_call_and_return_conditional_losses_52398
N__inference_blocks/b1/batchnorm_layer_call_and_return_conditional_losses_52416´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
/__inference_blocks/b1/prelu_layer_call_fn_48692Î
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
annotationsª *.¢+
)&"ÿÿÿÿÿÿÿÿÿ`ÿÿÿÿÿÿÿÿÿ
 2
J__inference_blocks/b1/prelu_layer_call_and_return_conditional_losses_48684Î
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
annotationsª *.¢+
)&"ÿÿÿÿÿÿÿÿÿ`ÿÿÿÿÿÿÿÿÿ
2
1__inference_blocks/b1/maxpool_layer_call_fn_48704à
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
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
´2±
L__inference_blocks/b1/maxpool_layer_call_and_return_conditional_losses_48698à
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
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ø2Õ
.__inference_blocks/b2/conv_layer_call_fn_52425¢
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
annotationsª *
 
ó2ð
I__inference_blocks/b2/conv_layer_call_and_return_conditional_losses_52435¢
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
annotationsª *
 
2
3__inference_blocks/b2/batchnorm_layer_call_fn_52448
3__inference_blocks/b2/batchnorm_layer_call_fn_52461
3__inference_blocks/b2/batchnorm_layer_call_fn_52474
3__inference_blocks/b2/batchnorm_layer_call_fn_52487´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ú2÷
N__inference_blocks/b2/batchnorm_layer_call_and_return_conditional_losses_52505
N__inference_blocks/b2/batchnorm_layer_call_and_return_conditional_losses_52523
N__inference_blocks/b2/batchnorm_layer_call_and_return_conditional_losses_52541
N__inference_blocks/b2/batchnorm_layer_call_and_return_conditional_losses_52559´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
/__inference_blocks/b2/prelu_layer_call_fn_49100Î
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
annotationsª *.¢+
)&"ÿÿÿÿÿÿÿÿÿ0ÿÿÿÿÿÿÿÿÿ
 2
J__inference_blocks/b2/prelu_layer_call_and_return_conditional_losses_49092Î
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
annotationsª *.¢+
)&"ÿÿÿÿÿÿÿÿÿ0ÿÿÿÿÿÿÿÿÿ
2
1__inference_blocks/b2/maxpool_layer_call_fn_49112à
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
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
´2±
L__inference_blocks/b2/maxpool_layer_call_and_return_conditional_losses_49106à
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
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ø2Õ
.__inference_blocks/b3/conv_layer_call_fn_52568¢
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
annotationsª *
 
ó2ð
I__inference_blocks/b3/conv_layer_call_and_return_conditional_losses_52578¢
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
annotationsª *
 
2
3__inference_blocks/b3/batchnorm_layer_call_fn_52591
3__inference_blocks/b3/batchnorm_layer_call_fn_52604
3__inference_blocks/b3/batchnorm_layer_call_fn_52617
3__inference_blocks/b3/batchnorm_layer_call_fn_52630´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ú2÷
N__inference_blocks/b3/batchnorm_layer_call_and_return_conditional_losses_52648
N__inference_blocks/b3/batchnorm_layer_call_and_return_conditional_losses_52666
N__inference_blocks/b3/batchnorm_layer_call_and_return_conditional_losses_52684
N__inference_blocks/b3/batchnorm_layer_call_and_return_conditional_losses_52702´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
/__inference_blocks/b3/prelu_layer_call_fn_49508Î
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
annotationsª *.¢+
)&"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 2
J__inference_blocks/b3/prelu_layer_call_and_return_conditional_losses_49500Î
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
annotationsª *.¢+
)&"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
1__inference_blocks/b3/maxpool_layer_call_fn_49520à
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
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
´2±
L__inference_blocks/b3/maxpool_layer_call_and_return_conditional_losses_49514à
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
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¹
 __inference__wrapped_model_48545 YZ[\]^_`abcdefghijklm01;8:9GMN5¢2
+¢(
&#
input_spÿÿÿÿÿÿÿÿÿ``
ª "9ª6
4
probability%"
probabilityÿÿÿÿÿÿÿÿÿé
N__inference_blocks/b1/batchnorm_layer_call_and_return_conditional_losses_52362[\]^M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 é
N__inference_blocks/b1/batchnorm_layer_call_and_return_conditional_losses_52380[\]^M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
N__inference_blocks/b1/batchnorm_layer_call_and_return_conditional_losses_52398r[\]^;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ``
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ``
 Ä
N__inference_blocks/b1/batchnorm_layer_call_and_return_conditional_losses_52416r[\]^;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ``
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ``
 Á
3__inference_blocks/b1/batchnorm_layer_call_fn_52305[\]^M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÁ
3__inference_blocks/b1/batchnorm_layer_call_fn_52318[\]^M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
3__inference_blocks/b1/batchnorm_layer_call_fn_52331e[\]^;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ``
p 
ª " ÿÿÿÿÿÿÿÿÿ``
3__inference_blocks/b1/batchnorm_layer_call_fn_52344e[\]^;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ``
p
ª " ÿÿÿÿÿÿÿÿÿ``¹
I__inference_blocks/b1/conv_layer_call_and_return_conditional_losses_52292lYZ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ``
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ``
 
.__inference_blocks/b1/conv_layer_call_fn_52282_YZ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ``
ª " ÿÿÿÿÿÿÿÿÿ``ï
L__inference_blocks/b1/maxpool_layer_call_and_return_conditional_losses_48698R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
1__inference_blocks/b1/maxpool_layer_call_fn_48704R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿË
J__inference_blocks/b1/prelu_layer_call_and_return_conditional_losses_48684}_@¢=
6¢3
1.
inputs"ÿÿÿÿÿÿÿÿÿ`ÿÿÿÿÿÿÿÿÿ
ª "6¢3
,)
0"ÿÿÿÿÿÿÿÿÿ`ÿÿÿÿÿÿÿÿÿ
 £
/__inference_blocks/b1/prelu_layer_call_fn_48692p_@¢=
6¢3
1.
inputs"ÿÿÿÿÿÿÿÿÿ`ÿÿÿÿÿÿÿÿÿ
ª ")&"ÿÿÿÿÿÿÿÿÿ`ÿÿÿÿÿÿÿÿÿÐ
D__inference_blocks/b1_layer_call_and_return_conditional_losses_48931YZ[\]^_M¢J
C¢@
63
blocks/b1/conv_inputÿÿÿÿÿÿÿÿÿ``
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ00
 Ð
D__inference_blocks/b1_layer_call_and_return_conditional_losses_48953YZ[\]^_M¢J
C¢@
63
blocks/b1/conv_inputÿÿÿÿÿÿÿÿÿ``
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ00
 Á
D__inference_blocks/b1_layer_call_and_return_conditional_losses_52032yYZ[\]^_?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ``
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ00
 Á
D__inference_blocks/b1_layer_call_and_return_conditional_losses_52065yYZ[\]^_?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ``
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ00
 §
)__inference_blocks/b1_layer_call_fn_48776zYZ[\]^_M¢J
C¢@
63
blocks/b1/conv_inputÿÿÿÿÿÿÿÿÿ``
p 

 
ª " ÿÿÿÿÿÿÿÿÿ00§
)__inference_blocks/b1_layer_call_fn_48909zYZ[\]^_M¢J
C¢@
63
blocks/b1/conv_inputÿÿÿÿÿÿÿÿÿ``
p

 
ª " ÿÿÿÿÿÿÿÿÿ00
)__inference_blocks/b1_layer_call_fn_51980lYZ[\]^_?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ``
p 

 
ª " ÿÿÿÿÿÿÿÿÿ00
)__inference_blocks/b1_layer_call_fn_51999lYZ[\]^_?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ``
p

 
ª " ÿÿÿÿÿÿÿÿÿ00é
N__inference_blocks/b2/batchnorm_layer_call_and_return_conditional_losses_52505bcdeM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 é
N__inference_blocks/b2/batchnorm_layer_call_and_return_conditional_losses_52523bcdeM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
N__inference_blocks/b2/batchnorm_layer_call_and_return_conditional_losses_52541rbcde;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ00
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ00
 Ä
N__inference_blocks/b2/batchnorm_layer_call_and_return_conditional_losses_52559rbcde;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ00
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ00
 Á
3__inference_blocks/b2/batchnorm_layer_call_fn_52448bcdeM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÁ
3__inference_blocks/b2/batchnorm_layer_call_fn_52461bcdeM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
3__inference_blocks/b2/batchnorm_layer_call_fn_52474ebcde;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ00
p 
ª " ÿÿÿÿÿÿÿÿÿ00
3__inference_blocks/b2/batchnorm_layer_call_fn_52487ebcde;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ00
p
ª " ÿÿÿÿÿÿÿÿÿ00¹
I__inference_blocks/b2/conv_layer_call_and_return_conditional_losses_52435l`a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ00
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ00
 
.__inference_blocks/b2/conv_layer_call_fn_52425_`a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ00
ª " ÿÿÿÿÿÿÿÿÿ00ï
L__inference_blocks/b2/maxpool_layer_call_and_return_conditional_losses_49106R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
1__inference_blocks/b2/maxpool_layer_call_fn_49112R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿË
J__inference_blocks/b2/prelu_layer_call_and_return_conditional_losses_49092}f@¢=
6¢3
1.
inputs"ÿÿÿÿÿÿÿÿÿ0ÿÿÿÿÿÿÿÿÿ
ª "6¢3
,)
0"ÿÿÿÿÿÿÿÿÿ0ÿÿÿÿÿÿÿÿÿ
 £
/__inference_blocks/b2/prelu_layer_call_fn_49100pf@¢=
6¢3
1.
inputs"ÿÿÿÿÿÿÿÿÿ0ÿÿÿÿÿÿÿÿÿ
ª ")&"ÿÿÿÿÿÿÿÿÿ0ÿÿÿÿÿÿÿÿÿÐ
D__inference_blocks/b2_layer_call_and_return_conditional_losses_49339`abcdefM¢J
C¢@
63
blocks/b2/conv_inputÿÿÿÿÿÿÿÿÿ00
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Ð
D__inference_blocks/b2_layer_call_and_return_conditional_losses_49361`abcdefM¢J
C¢@
63
blocks/b2/conv_inputÿÿÿÿÿÿÿÿÿ00
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Á
D__inference_blocks/b2_layer_call_and_return_conditional_losses_52136y`abcdef?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ00
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Á
D__inference_blocks/b2_layer_call_and_return_conditional_losses_52169y`abcdef?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ00
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 §
)__inference_blocks/b2_layer_call_fn_49184z`abcdefM¢J
C¢@
63
blocks/b2/conv_inputÿÿÿÿÿÿÿÿÿ00
p 

 
ª " ÿÿÿÿÿÿÿÿÿ§
)__inference_blocks/b2_layer_call_fn_49317z`abcdefM¢J
C¢@
63
blocks/b2/conv_inputÿÿÿÿÿÿÿÿÿ00
p

 
ª " ÿÿÿÿÿÿÿÿÿ
)__inference_blocks/b2_layer_call_fn_52084l`abcdef?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ00
p 

 
ª " ÿÿÿÿÿÿÿÿÿ
)__inference_blocks/b2_layer_call_fn_52103l`abcdef?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ00
p

 
ª " ÿÿÿÿÿÿÿÿÿé
N__inference_blocks/b3/batchnorm_layer_call_and_return_conditional_losses_52648ijklM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 é
N__inference_blocks/b3/batchnorm_layer_call_and_return_conditional_losses_52666ijklM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
N__inference_blocks/b3/batchnorm_layer_call_and_return_conditional_losses_52684rijkl;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Ä
N__inference_blocks/b3/batchnorm_layer_call_and_return_conditional_losses_52702rijkl;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Á
3__inference_blocks/b3/batchnorm_layer_call_fn_52591ijklM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÁ
3__inference_blocks/b3/batchnorm_layer_call_fn_52604ijklM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
3__inference_blocks/b3/batchnorm_layer_call_fn_52617eijkl;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª " ÿÿÿÿÿÿÿÿÿ
3__inference_blocks/b3/batchnorm_layer_call_fn_52630eijkl;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª " ÿÿÿÿÿÿÿÿÿ¹
I__inference_blocks/b3/conv_layer_call_and_return_conditional_losses_52578lgh7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
.__inference_blocks/b3/conv_layer_call_fn_52568_gh7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿï
L__inference_blocks/b3/maxpool_layer_call_and_return_conditional_losses_49514R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
1__inference_blocks/b3/maxpool_layer_call_fn_49520R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿË
J__inference_blocks/b3/prelu_layer_call_and_return_conditional_losses_49500}m@¢=
6¢3
1.
inputs"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "6¢3
,)
0"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 £
/__inference_blocks/b3/prelu_layer_call_fn_49508pm@¢=
6¢3
1.
inputs"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ")&"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÐ
D__inference_blocks/b3_layer_call_and_return_conditional_losses_49747ghijklmM¢J
C¢@
63
blocks/b3/conv_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Ð
D__inference_blocks/b3_layer_call_and_return_conditional_losses_49769ghijklmM¢J
C¢@
63
blocks/b3/conv_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Á
D__inference_blocks/b3_layer_call_and_return_conditional_losses_52240yghijklm?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Á
D__inference_blocks/b3_layer_call_and_return_conditional_losses_52273yghijklm?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 §
)__inference_blocks/b3_layer_call_fn_49592zghijklmM¢J
C¢@
63
blocks/b3/conv_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª " ÿÿÿÿÿÿÿÿÿ§
)__inference_blocks/b3_layer_call_fn_49725zghijklmM¢J
C¢@
63
blocks/b3/conv_inputÿÿÿÿÿÿÿÿÿ
p

 
ª " ÿÿÿÿÿÿÿÿÿ
)__inference_blocks/b3_layer_call_fn_52188lghijklm?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª " ÿÿÿÿÿÿÿÿÿ
)__inference_blocks/b3_layer_call_fn_52207lghijklm?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª " ÿÿÿÿÿÿÿÿÿÓ
A__inference_blocks_layer_call_and_return_conditional_losses_50106YZ[\]^_`abcdefghijklmE¢B
;¢8
.+
blocks/inputÿÿÿÿÿÿÿÿÿ``
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Ó
A__inference_blocks_layer_call_and_return_conditional_losses_50155YZ[\]^_`abcdefghijklmE¢B
;¢8
.+
blocks/inputÿÿÿÿÿÿÿÿÿ``
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Í
A__inference_blocks_layer_call_and_return_conditional_losses_51733YZ[\]^_`abcdefghijklm?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ``
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Í
A__inference_blocks_layer_call_and_return_conditional_losses_51824YZ[\]^_`abcdefghijklm?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ``
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 «
&__inference_blocks_layer_call_fn_49867YZ[\]^_`abcdefghijklmE¢B
;¢8
.+
blocks/inputÿÿÿÿÿÿÿÿÿ``
p 

 
ª " ÿÿÿÿÿÿÿÿÿ«
&__inference_blocks_layer_call_fn_50057YZ[\]^_`abcdefghijklmE¢B
;¢8
.+
blocks/inputÿÿÿÿÿÿÿÿÿ``
p

 
ª " ÿÿÿÿÿÿÿÿÿ¤
&__inference_blocks_layer_call_fn_51595zYZ[\]^_`abcdefghijklm?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ``
p 

 
ª " ÿÿÿÿÿÿÿÿÿ¤
&__inference_blocks_layer_call_fn_51642zYZ[\]^_`abcdefghijklm?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ``
p

 
ª " ÿÿÿÿÿÿÿÿÿ 
@__inference_dense_layer_call_and_return_conditional_losses_51843\01/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 x
%__inference_dense_layer_call_fn_51833O01/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
D__inference_dense_out_layer_call_and_return_conditional_losses_51961\MN/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dense_out_layer_call_fn_51951OMN/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ­
G__inference_fc_batchnorm_layer_call_and_return_conditional_losses_51889b;8:93¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ­
G__inference_fc_batchnorm_layer_call_and_return_conditional_losses_51923b:;893¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_fc_batchnorm_layer_call_fn_51856U;8:93¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_fc_batchnorm_layer_call_fn_51869U:;893¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¥
E__inference_fc_dropout_layer_call_and_return_conditional_losses_51938\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¥
E__inference_fc_dropout_layer_call_and_return_conditional_losses_51942\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_fc_dropout_layer_call_fn_51928O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ}
*__inference_fc_dropout_layer_call_fn_51933O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ´
C__inference_fc_prelu_layer_call_and_return_conditional_losses_50343mG8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
(__inference_fc_prelu_layer_call_fn_50351`G8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÑ
H__inference_globalavgpool_layer_call_and_return_conditional_losses_50162R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¨
-__inference_globalavgpool_layer_call_fn_50168wR¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¶
N__inference_insert_channel_axis_layer_call_and_return_conditional_losses_43396d3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ``
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ``
 
3__inference_insert_channel_axis_layer_call_fn_42951W3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ``
ª " ÿÿÿÿÿÿÿÿÿ``¢
F__inference_probability_layer_call_and_return_conditional_losses_43277X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
+__inference_probability_layer_call_fn_42553K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÈ
#__inference_signature_wrapper_51107  YZ[\]^_`abcdefghijklm01;8:9GMNA¢>
¢ 
7ª4
2
input_sp&#
input_spÿÿÿÿÿÿÿÿÿ``"9ª6
4
probability%"
probabilityÿÿÿÿÿÿÿÿÿÑ
D__inference_simplecnn_layer_call_and_return_conditional_losses_50950 YZ[\]^_`abcdefghijklm01;8:9GMN=¢:
3¢0
&#
input_spÿÿÿÿÿÿÿÿÿ``
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ñ
D__inference_simplecnn_layer_call_and_return_conditional_losses_51036 YZ[\]^_`abcdefghijklm01:;89GMN=¢:
3¢0
&#
input_spÿÿÿÿÿÿÿÿÿ``
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ï
D__inference_simplecnn_layer_call_and_return_conditional_losses_51390 YZ[\]^_`abcdefghijklm01;8:9GMN;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ``
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ï
D__inference_simplecnn_layer_call_and_return_conditional_losses_51548 YZ[\]^_`abcdefghijklm01:;89GMN;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ``
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¨
)__inference_simplecnn_layer_call_fn_50536{ YZ[\]^_`abcdefghijklm01;8:9GMN=¢:
3¢0
&#
input_spÿÿÿÿÿÿÿÿÿ``
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¨
)__inference_simplecnn_layer_call_fn_50864{ YZ[\]^_`abcdefghijklm01:;89GMN=¢:
3¢0
&#
input_spÿÿÿÿÿÿÿÿÿ``
p

 
ª "ÿÿÿÿÿÿÿÿÿ¦
)__inference_simplecnn_layer_call_fn_51176y YZ[\]^_`abcdefghijklm01;8:9GMN;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ``
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¦
)__inference_simplecnn_layer_call_fn_51245y YZ[\]^_`abcdefghijklm01:;89GMN;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ``
p

 
ª "ÿÿÿÿÿÿÿÿÿ