#
Ó
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
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

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
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
E
Relu
features"T
activations"T"
Ttype:
2	
¥
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
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
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Á
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
executor_typestring ¨
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
-
Tanh
x"T
y"T"
Ttype:

2
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68þç!

embedding_4/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Äd*'
shared_nameembedding_4/embeddings

*embedding_4/embeddings/Read/ReadVariableOpReadVariableOpembedding_4/embeddings*
_output_shapes
:	Äd*
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:@ *
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
: *
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

: *
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

lstm_2/lstm_cell_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d**
shared_namelstm_2/lstm_cell_3/kernel

-lstm_2/lstm_cell_3/kernel/Read/ReadVariableOpReadVariableOplstm_2/lstm_cell_3/kernel*
_output_shapes
:	d*
dtype0
£
#lstm_2/lstm_cell_3/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*4
shared_name%#lstm_2/lstm_cell_3/recurrent_kernel

7lstm_2/lstm_cell_3/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_2/lstm_cell_3/recurrent_kernel*
_output_shapes
:	@*
dtype0

lstm_2/lstm_cell_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namelstm_2/lstm_cell_3/bias

+lstm_2/lstm_cell_3/bias/Read/ReadVariableOpReadVariableOplstm_2/lstm_cell_3/bias*
_output_shapes	
:*
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

Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes

:@ *
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
: *
dtype0

Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_5/kernel/m

)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes

: *
dtype0
~
Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/m
w
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes
:*
dtype0

 Adam/lstm_2/lstm_cell_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*1
shared_name" Adam/lstm_2/lstm_cell_3/kernel/m

4Adam/lstm_2/lstm_cell_3/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_2/lstm_cell_3/kernel/m*
_output_shapes
:	d*
dtype0
±
*Adam/lstm_2/lstm_cell_3/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*;
shared_name,*Adam/lstm_2/lstm_cell_3/recurrent_kernel/m
ª
>Adam/lstm_2/lstm_cell_3/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_2/lstm_cell_3/recurrent_kernel/m*
_output_shapes
:	@*
dtype0

Adam/lstm_2/lstm_cell_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/lstm_2/lstm_cell_3/bias/m

2Adam/lstm_2/lstm_cell_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_2/lstm_cell_3/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes

:@ *
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
: *
dtype0

Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_5/kernel/v

)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes

: *
dtype0
~
Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/v
w
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes
:*
dtype0

 Adam/lstm_2/lstm_cell_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*1
shared_name" Adam/lstm_2/lstm_cell_3/kernel/v

4Adam/lstm_2/lstm_cell_3/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_2/lstm_cell_3/kernel/v*
_output_shapes
:	d*
dtype0
±
*Adam/lstm_2/lstm_cell_3/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*;
shared_name,*Adam/lstm_2/lstm_cell_3/recurrent_kernel/v
ª
>Adam/lstm_2/lstm_cell_3/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_2/lstm_cell_3/recurrent_kernel/v*
_output_shapes
:	@*
dtype0

Adam/lstm_2/lstm_cell_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/lstm_2/lstm_cell_3/bias/v

2Adam/lstm_2/lstm_cell_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_2/lstm_cell_3/bias/v*
_output_shapes	
:*
dtype0

NoOpNoOp
º:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*õ9
valueë9Bè9 Bá9
è
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
 

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
Á
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses*
¦

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses*
Â
.iter

/beta_1

0beta_2
	1decay
2learning_ratemimj&mk'ml3mm4mn5movpvq&vr'vs3vt4vu5vv*
<
0
31
42
53
4
5
&6
'7*
5
30
41
52
3
4
&5
'6*
* 
°
6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

;serving_default* 
jd
VARIABLE_VALUEembedding_4/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

0*
* 
* 

<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
ã
A
state_size

3kernel
4recurrent_kernel
5bias
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F_random_generator
G__call__
*H&call_and_return_all_conditional_losses*
* 

30
41
52*

30
41
52*
* 


Istates
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
^X
VARIABLE_VALUEdense_4/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_4/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_5/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_5/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

&0
'1*

&0
'1*
* 

Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_2/lstm_cell_3/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#lstm_2/lstm_cell_3/recurrent_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUElstm_2/lstm_cell_3/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*

0*
 
0
1
2
3*

Y0
Z1*
* 
* 
* 

0*
* 
* 
* 
* 
* 

30
41
52*

30
41
52*
* 

[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	`total
	acount
b	variables
c	keras_api*
H
	dtotal
	ecount
f
_fn_kwargs
g	variables
h	keras_api*
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

`0
a1*

b	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

d0
e1*

g	variables*
{
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_5/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_5/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_2/lstm_cell_3/kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/lstm_2/lstm_cell_3/recurrent_kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/lstm_2/lstm_cell_3/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_5/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_5/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_2/lstm_cell_3/kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/lstm_2/lstm_cell_3/recurrent_kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/lstm_2/lstm_cell_3/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

!serving_default_embedding_4_inputPlaceholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿú
ø
StatefulPartitionedCallStatefulPartitionedCall!serving_default_embedding_4_inputembedding_4/embeddingslstm_2/lstm_cell_3/kernellstm_2/lstm_cell_3/bias#lstm_2/lstm_cell_3/recurrent_kerneldense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_23869
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ã
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*embedding_4/embeddings/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-lstm_2/lstm_cell_3/kernel/Read/ReadVariableOp7lstm_2/lstm_cell_3/recurrent_kernel/Read/ReadVariableOp+lstm_2/lstm_cell_3/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp4Adam/lstm_2/lstm_cell_3/kernel/m/Read/ReadVariableOp>Adam/lstm_2/lstm_cell_3/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_2/lstm_cell_3/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOp4Adam/lstm_2/lstm_cell_3/kernel/v/Read/ReadVariableOp>Adam/lstm_2/lstm_cell_3/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_2/lstm_cell_3/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_25576
Ò
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_4/embeddingsdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_2/lstm_cell_3/kernel#lstm_2/lstm_cell_3/recurrent_kernellstm_2/lstm_cell_3/biastotalcounttotal_1count_1Adam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense_5/kernel/mAdam/dense_5/bias/m Adam/lstm_2/lstm_cell_3/kernel/m*Adam/lstm_2/lstm_cell_3/recurrent_kernel/mAdam/lstm_2/lstm_cell_3/bias/mAdam/dense_4/kernel/vAdam/dense_4/bias/vAdam/dense_5/kernel/vAdam/dense_5/bias/v Adam/lstm_2/lstm_cell_3/kernel/v*Adam/lstm_2/lstm_cell_3/recurrent_kernel/vAdam/lstm_2/lstm_cell_3/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_25679ùÕ 
Ñ
ò
G__inference_sequential_4_layer_call_and_return_conditional_losses_22569

inputs$
embedding_4_22281:	Äd
lstm_2_22527:	d
lstm_2_22529:	
lstm_2_22531:	@
dense_4_22546:@ 
dense_4_22548: 
dense_5_22563: 
dense_5_22565:
identity¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢#embedding_4/StatefulPartitionedCall¢lstm_2/StatefulPartitionedCallé
#embedding_4/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_4_22281*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúd*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_embedding_4_layer_call_and_return_conditional_losses_22280
lstm_2/StatefulPartitionedCallStatefulPartitionedCall,embedding_4/StatefulPartitionedCall:output:0lstm_2_22527lstm_2_22529lstm_2_22531*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_22526
dense_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0dense_4_22546dense_4_22548*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_22545
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_22563dense_5_22565*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_22562w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
NoOpNoOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall$^embedding_4/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿú: : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2J
#embedding_4/StatefulPartitionedCall#embedding_4/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs
¶Ç
Ü
A__inference_lstm_2_layer_call_and_return_conditional_losses_22992

inputs<
)lstm_cell_3_split_readvariableop_resource:	d:
+lstm_cell_3_split_1_readvariableop_resource:	6
#lstm_cell_3_readvariableop_resource:	@
identity¢lstm_cell_3/ReadVariableOp¢lstm_cell_3/ReadVariableOp_1¢lstm_cell_3/ReadVariableOp_2¢lstm_cell_3/ReadVariableOp_3¢ lstm_cell_3/split/ReadVariableOp¢"lstm_cell_3/split_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:úÿÿÿÿÿÿÿÿÿdD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskc
lstm_cell_3/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:`
lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_3/ones_likeFill$lstm_cell_3/ones_like/Shape:output:0$lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd^
lstm_cell_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_3/dropout/MulMullstm_cell_3/ones_like:output:0"lstm_cell_3/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
lstm_cell_3/dropout/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:¤
0lstm_cell_3/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_3/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0g
"lstm_cell_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ê
 lstm_cell_3/dropout/GreaterEqualGreaterEqual9lstm_cell_3/dropout/random_uniform/RandomUniform:output:0+lstm_cell_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/dropout/CastCast$lstm_cell_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/dropout/Mul_1Mullstm_cell_3/dropout/Mul:z:0lstm_cell_3/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
lstm_cell_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_3/dropout_1/MulMullstm_cell_3/ones_like:output:0$lstm_cell_3/dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
lstm_cell_3/dropout_1/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:¨
2lstm_cell_3/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0i
$lstm_cell_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ð
"lstm_cell_3/dropout_1/GreaterEqualGreaterEqual;lstm_cell_3/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/dropout_1/CastCast&lstm_cell_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/dropout_1/Mul_1Mullstm_cell_3/dropout_1/Mul:z:0lstm_cell_3/dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
lstm_cell_3/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_3/dropout_2/MulMullstm_cell_3/ones_like:output:0$lstm_cell_3/dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
lstm_cell_3/dropout_2/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:¨
2lstm_cell_3/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0i
$lstm_cell_3/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ð
"lstm_cell_3/dropout_2/GreaterEqualGreaterEqual;lstm_cell_3/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/dropout_2/CastCast&lstm_cell_3/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/dropout_2/Mul_1Mullstm_cell_3/dropout_2/Mul:z:0lstm_cell_3/dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
lstm_cell_3/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_3/dropout_3/MulMullstm_cell_3/ones_like:output:0$lstm_cell_3/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
lstm_cell_3/dropout_3/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:¨
2lstm_cell_3/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0i
$lstm_cell_3/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ð
"lstm_cell_3/dropout_3/GreaterEqualGreaterEqual;lstm_cell_3/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/dropout_3/CastCast&lstm_cell_3/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/dropout_3/Mul_1Mullstm_cell_3/dropout_3/Mul:z:0lstm_cell_3/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
lstm_cell_3/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:b
lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
lstm_cell_3/ones_like_1Fill&lstm_cell_3/ones_like_1/Shape:output:0&lstm_cell_3/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
lstm_cell_3/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_3/dropout_4/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
lstm_cell_3/dropout_4/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:¨
2lstm_cell_3/dropout_4/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0i
$lstm_cell_3/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ð
"lstm_cell_3/dropout_4/GreaterEqualGreaterEqual;lstm_cell_3/dropout_4/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/dropout_4/CastCast&lstm_cell_3/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/dropout_4/Mul_1Mullstm_cell_3/dropout_4/Mul:z:0lstm_cell_3/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
lstm_cell_3/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_3/dropout_5/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
lstm_cell_3/dropout_5/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:¨
2lstm_cell_3/dropout_5/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0i
$lstm_cell_3/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ð
"lstm_cell_3/dropout_5/GreaterEqualGreaterEqual;lstm_cell_3/dropout_5/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/dropout_5/CastCast&lstm_cell_3/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/dropout_5/Mul_1Mullstm_cell_3/dropout_5/Mul:z:0lstm_cell_3/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
lstm_cell_3/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_3/dropout_6/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_6/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
lstm_cell_3/dropout_6/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:¨
2lstm_cell_3/dropout_6/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_6/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0i
$lstm_cell_3/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ð
"lstm_cell_3/dropout_6/GreaterEqualGreaterEqual;lstm_cell_3/dropout_6/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/dropout_6/CastCast&lstm_cell_3/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/dropout_6/Mul_1Mullstm_cell_3/dropout_6/Mul:z:0lstm_cell_3/dropout_6/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
lstm_cell_3/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_3/dropout_7/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_7/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
lstm_cell_3/dropout_7/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:¨
2lstm_cell_3/dropout_7/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_7/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0i
$lstm_cell_3/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ð
"lstm_cell_3/dropout_7/GreaterEqualGreaterEqual;lstm_cell_3/dropout_7/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/dropout_7/CastCast&lstm_cell_3/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/dropout_7/Mul_1Mullstm_cell_3/dropout_7/Mul:z:0lstm_cell_3/dropout_7/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/mulMulstrided_slice_2:output:0lstm_cell_3/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/mul_1Mulstrided_slice_2:output:0lstm_cell_3/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/mul_2Mulstrided_slice_2:output:0lstm_cell_3/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/mul_3Mulstrided_slice_2:output:0lstm_cell_3/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_cell_3/split/ReadVariableOpReadVariableOp)lstm_cell_3_split_readvariableop_resource*
_output_shapes
:	d*
dtype0Â
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0(lstm_cell_3/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:d@:d@:d@:d@*
	num_split
lstm_cell_3/MatMulMatMullstm_cell_3/mul:z:0lstm_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/MatMul_1MatMullstm_cell_3/mul_1:z:0lstm_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/MatMul_2MatMullstm_cell_3/mul_2:z:0lstm_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/MatMul_3MatMullstm_cell_3/mul_3:z:0lstm_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
"lstm_cell_3/split_1/ReadVariableOpReadVariableOp+lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0¸
lstm_cell_3/split_1Split&lstm_cell_3/split_1/split_dim:output:0*lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split
lstm_cell_3/BiasAddBiasAddlstm_cell_3/MatMul:product:0lstm_cell_3/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/BiasAdd_1BiasAddlstm_cell_3/MatMul_1:product:0lstm_cell_3/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/BiasAdd_2BiasAddlstm_cell_3/MatMul_2:product:0lstm_cell_3/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/BiasAdd_3BiasAddlstm_cell_3/MatMul_3:product:0lstm_cell_3/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
lstm_cell_3/mul_4Mulzeros:output:0lstm_cell_3/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
lstm_cell_3/mul_5Mulzeros:output:0lstm_cell_3/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
lstm_cell_3/mul_6Mulzeros:output:0lstm_cell_3/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
lstm_cell_3/mul_7Mulzeros:output:0lstm_cell_3/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/ReadVariableOpReadVariableOp#lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0p
lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   r
!lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
lstm_cell_3/strided_sliceStridedSlice"lstm_cell_3/ReadVariableOp:value:0(lstm_cell_3/strided_slice/stack:output:0*lstm_cell_3/strided_slice/stack_1:output:0*lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
lstm_cell_3/MatMul_4MatMullstm_cell_3/mul_4:z:0"lstm_cell_3/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/addAddV2lstm_cell_3/BiasAdd:output:0lstm_cell_3/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
lstm_cell_3/SigmoidSigmoidlstm_cell_3/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/ReadVariableOp_1ReadVariableOp#lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0r
!lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   t
#lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ±
lstm_cell_3/strided_slice_1StridedSlice$lstm_cell_3/ReadVariableOp_1:value:0*lstm_cell_3/strided_slice_1/stack:output:0,lstm_cell_3/strided_slice_1/stack_1:output:0,lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
lstm_cell_3/MatMul_5MatMullstm_cell_3/mul_5:z:0$lstm_cell_3/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/add_1AddV2lstm_cell_3/BiasAdd_1:output:0lstm_cell_3/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
lstm_cell_3/mul_8Mullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/ReadVariableOp_2ReadVariableOp#lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0r
!lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   t
#lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ±
lstm_cell_3/strided_slice_2StridedSlice$lstm_cell_3/ReadVariableOp_2:value:0*lstm_cell_3/strided_slice_2/stack:output:0,lstm_cell_3/strided_slice_2/stack_1:output:0,lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
lstm_cell_3/MatMul_6MatMullstm_cell_3/mul_6:z:0$lstm_cell_3/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/add_2AddV2lstm_cell_3/BiasAdd_2:output:0lstm_cell_3/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
lstm_cell_3/TanhTanhlstm_cell_3/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
lstm_cell_3/mul_9Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
lstm_cell_3/add_3AddV2lstm_cell_3/mul_8:z:0lstm_cell_3/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/ReadVariableOp_3ReadVariableOp#lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0r
!lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   t
#lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ±
lstm_cell_3/strided_slice_3StridedSlice$lstm_cell_3/ReadVariableOp_3:value:0*lstm_cell_3/strided_slice_3/stack:output:0,lstm_cell_3/strided_slice_3/stack_1:output:0,lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
lstm_cell_3/MatMul_7MatMullstm_cell_3/mul_7:z:0$lstm_cell_3/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/add_4AddV2lstm_cell_3/BiasAdd_3:output:0lstm_cell_3/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
lstm_cell_3/Tanh_1Tanhlstm_cell_3/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
lstm_cell_3/mul_10Mullstm_cell_3/Sigmoid_2:y:0lstm_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ó
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_3_split_readvariableop_resource+lstm_cell_3_split_1_readvariableop_resource#lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_22794*
condR
while_cond_22793*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ã
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:úÿÿÿÿÿÿÿÿÿ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿú@[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^lstm_cell_3/ReadVariableOp^lstm_cell_3/ReadVariableOp_1^lstm_cell_3/ReadVariableOp_2^lstm_cell_3/ReadVariableOp_3!^lstm_cell_3/split/ReadVariableOp#^lstm_cell_3/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿúd: : : 28
lstm_cell_3/ReadVariableOplstm_cell_3/ReadVariableOp2<
lstm_cell_3/ReadVariableOp_1lstm_cell_3/ReadVariableOp_12<
lstm_cell_3/ReadVariableOp_2lstm_cell_3/ReadVariableOp_22<
lstm_cell_3/ReadVariableOp_3lstm_cell_3/ReadVariableOp_32D
 lstm_cell_3/split/ReadVariableOp lstm_cell_3/split/ReadVariableOp2H
"lstm_cell_3/split_1/ReadVariableOp"lstm_cell_3/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúd
 
_user_specified_nameinputs
°
¾
while_cond_24345
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_24345___redundant_placeholder03
/while_while_cond_24345___redundant_placeholder13
/while_while_cond_24345___redundant_placeholder23
/while_while_cond_24345___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ç
ô
+__inference_lstm_cell_3_layer_call_fn_25232

inputs
states_0
states_1
unknown:	d
	unknown_0:	
	unknown_1:	@
identity

identity_1

identity_2¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_22126o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/1
Î	
Ã
,__inference_sequential_4_layer_call_fn_23171

inputs
unknown:	Äd
	unknown_0:	d
	unknown_1:	
	unknown_2:	@
	unknown_3:@ 
	unknown_4: 
	unknown_5: 
	unknown_6:
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_22569o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿú: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs


ó
B__inference_dense_5_layer_call_and_return_conditional_losses_25198

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
°
¾
while_cond_24038
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_24038___redundant_placeholder03
/while_while_cond_24038___redundant_placeholder13
/while_while_cond_24038___redundant_placeholder23
/while_while_cond_24038___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ö
â
G__inference_sequential_4_layer_call_and_return_conditional_losses_23846

inputs5
"embedding_4_embedding_lookup_23459:	ÄdC
0lstm_2_lstm_cell_3_split_readvariableop_resource:	dA
2lstm_2_lstm_cell_3_split_1_readvariableop_resource:	=
*lstm_2_lstm_cell_3_readvariableop_resource:	@8
&dense_4_matmul_readvariableop_resource:@ 5
'dense_4_biasadd_readvariableop_resource: 8
&dense_5_matmul_readvariableop_resource: 5
'dense_5_biasadd_readvariableop_resource:
identity¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢dense_5/BiasAdd/ReadVariableOp¢dense_5/MatMul/ReadVariableOp¢embedding_4/embedding_lookup¢!lstm_2/lstm_cell_3/ReadVariableOp¢#lstm_2/lstm_cell_3/ReadVariableOp_1¢#lstm_2/lstm_cell_3/ReadVariableOp_2¢#lstm_2/lstm_cell_3/ReadVariableOp_3¢'lstm_2/lstm_cell_3/split/ReadVariableOp¢)lstm_2/lstm_cell_3/split_1/ReadVariableOp¢lstm_2/whileb
embedding_4/CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿúê
embedding_4/embedding_lookupResourceGather"embedding_4_embedding_lookup_23459embedding_4/Cast:y:0*
Tindices0*5
_class+
)'loc:@embedding_4/embedding_lookup/23459*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúd*
dtype0Æ
%embedding_4/embedding_lookup/IdentityIdentity%embedding_4/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_4/embedding_lookup/23459*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúd
'embedding_4/embedding_lookup/Identity_1Identity.embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúdl
lstm_2/ShapeShape0embedding_4/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:d
lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
lstm_2/strided_sliceStridedSlicelstm_2/Shape:output:0#lstm_2/strided_slice/stack:output:0%lstm_2/strided_slice/stack_1:output:0%lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@
lstm_2/zeros/packedPacklstm_2/strided_slice:output:0lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_2/zerosFilllstm_2/zeros/packed:output:0lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@
lstm_2/zeros_1/packedPacklstm_2/strided_slice:output:0 lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_2/zeros_1Filllstm_2/zeros_1/packed:output:0lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@j
lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¦
lstm_2/transpose	Transpose0embedding_4/embedding_lookup/Identity_1:output:0lstm_2/transpose/perm:output:0*
T0*,
_output_shapes
:úÿÿÿÿÿÿÿÿÿdR
lstm_2/Shape_1Shapelstm_2/transpose:y:0*
T0*
_output_shapes
:f
lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
lstm_2/strided_slice_1StridedSlicelstm_2/Shape_1:output:0%lstm_2/strided_slice_1/stack:output:0'lstm_2/strided_slice_1/stack_1:output:0'lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÉ
lstm_2/TensorArrayV2TensorListReserve+lstm_2/TensorArrayV2/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   õ
.lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_2/transpose:y:0Elstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒf
lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_2/strided_slice_2StridedSlicelstm_2/transpose:y:0%lstm_2/strided_slice_2/stack:output:0'lstm_2/strided_slice_2/stack_1:output:0'lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskq
"lstm_2/lstm_cell_3/ones_like/ShapeShapelstm_2/strided_slice_2:output:0*
T0*
_output_shapes
:g
"lstm_2/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
lstm_2/lstm_cell_3/ones_likeFill+lstm_2/lstm_cell_3/ones_like/Shape:output:0+lstm_2/lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿde
 lstm_2/lstm_cell_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?©
lstm_2/lstm_cell_3/dropout/MulMul%lstm_2/lstm_cell_3/ones_like:output:0)lstm_2/lstm_cell_3/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
 lstm_2/lstm_cell_3/dropout/ShapeShape%lstm_2/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:²
7lstm_2/lstm_cell_3/dropout/random_uniform/RandomUniformRandomUniform)lstm_2/lstm_cell_3/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0n
)lstm_2/lstm_cell_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ß
'lstm_2/lstm_cell_3/dropout/GreaterEqualGreaterEqual@lstm_2/lstm_cell_3/dropout/random_uniform/RandomUniform:output:02lstm_2/lstm_cell_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_2/lstm_cell_3/dropout/CastCast+lstm_2/lstm_cell_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
 lstm_2/lstm_cell_3/dropout/Mul_1Mul"lstm_2/lstm_cell_3/dropout/Mul:z:0#lstm_2/lstm_cell_3/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
"lstm_2/lstm_cell_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?­
 lstm_2/lstm_cell_3/dropout_1/MulMul%lstm_2/lstm_cell_3/ones_like:output:0+lstm_2/lstm_cell_3/dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
"lstm_2/lstm_cell_3/dropout_1/ShapeShape%lstm_2/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:¶
9lstm_2/lstm_cell_3/dropout_1/random_uniform/RandomUniformRandomUniform+lstm_2/lstm_cell_3/dropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0p
+lstm_2/lstm_cell_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)lstm_2/lstm_cell_3/dropout_1/GreaterEqualGreaterEqualBlstm_2/lstm_cell_3/dropout_1/random_uniform/RandomUniform:output:04lstm_2/lstm_cell_3/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!lstm_2/lstm_cell_3/dropout_1/CastCast-lstm_2/lstm_cell_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¨
"lstm_2/lstm_cell_3/dropout_1/Mul_1Mul$lstm_2/lstm_cell_3/dropout_1/Mul:z:0%lstm_2/lstm_cell_3/dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
"lstm_2/lstm_cell_3/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?­
 lstm_2/lstm_cell_3/dropout_2/MulMul%lstm_2/lstm_cell_3/ones_like:output:0+lstm_2/lstm_cell_3/dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
"lstm_2/lstm_cell_3/dropout_2/ShapeShape%lstm_2/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:¶
9lstm_2/lstm_cell_3/dropout_2/random_uniform/RandomUniformRandomUniform+lstm_2/lstm_cell_3/dropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0p
+lstm_2/lstm_cell_3/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)lstm_2/lstm_cell_3/dropout_2/GreaterEqualGreaterEqualBlstm_2/lstm_cell_3/dropout_2/random_uniform/RandomUniform:output:04lstm_2/lstm_cell_3/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!lstm_2/lstm_cell_3/dropout_2/CastCast-lstm_2/lstm_cell_3/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¨
"lstm_2/lstm_cell_3/dropout_2/Mul_1Mul$lstm_2/lstm_cell_3/dropout_2/Mul:z:0%lstm_2/lstm_cell_3/dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
"lstm_2/lstm_cell_3/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?­
 lstm_2/lstm_cell_3/dropout_3/MulMul%lstm_2/lstm_cell_3/ones_like:output:0+lstm_2/lstm_cell_3/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
"lstm_2/lstm_cell_3/dropout_3/ShapeShape%lstm_2/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:¶
9lstm_2/lstm_cell_3/dropout_3/random_uniform/RandomUniformRandomUniform+lstm_2/lstm_cell_3/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0p
+lstm_2/lstm_cell_3/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)lstm_2/lstm_cell_3/dropout_3/GreaterEqualGreaterEqualBlstm_2/lstm_cell_3/dropout_3/random_uniform/RandomUniform:output:04lstm_2/lstm_cell_3/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!lstm_2/lstm_cell_3/dropout_3/CastCast-lstm_2/lstm_cell_3/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¨
"lstm_2/lstm_cell_3/dropout_3/Mul_1Mul$lstm_2/lstm_cell_3/dropout_3/Mul:z:0%lstm_2/lstm_cell_3/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
$lstm_2/lstm_cell_3/ones_like_1/ShapeShapelstm_2/zeros:output:0*
T0*
_output_shapes
:i
$lstm_2/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
lstm_2/lstm_cell_3/ones_like_1Fill-lstm_2/lstm_cell_3/ones_like_1/Shape:output:0-lstm_2/lstm_cell_3/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
"lstm_2/lstm_cell_3/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¯
 lstm_2/lstm_cell_3/dropout_4/MulMul'lstm_2/lstm_cell_3/ones_like_1:output:0+lstm_2/lstm_cell_3/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
"lstm_2/lstm_cell_3/dropout_4/ShapeShape'lstm_2/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:¶
9lstm_2/lstm_cell_3/dropout_4/random_uniform/RandomUniformRandomUniform+lstm_2/lstm_cell_3/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0p
+lstm_2/lstm_cell_3/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)lstm_2/lstm_cell_3/dropout_4/GreaterEqualGreaterEqualBlstm_2/lstm_cell_3/dropout_4/random_uniform/RandomUniform:output:04lstm_2/lstm_cell_3/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
!lstm_2/lstm_cell_3/dropout_4/CastCast-lstm_2/lstm_cell_3/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¨
"lstm_2/lstm_cell_3/dropout_4/Mul_1Mul$lstm_2/lstm_cell_3/dropout_4/Mul:z:0%lstm_2/lstm_cell_3/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
"lstm_2/lstm_cell_3/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¯
 lstm_2/lstm_cell_3/dropout_5/MulMul'lstm_2/lstm_cell_3/ones_like_1:output:0+lstm_2/lstm_cell_3/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
"lstm_2/lstm_cell_3/dropout_5/ShapeShape'lstm_2/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:¶
9lstm_2/lstm_cell_3/dropout_5/random_uniform/RandomUniformRandomUniform+lstm_2/lstm_cell_3/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0p
+lstm_2/lstm_cell_3/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)lstm_2/lstm_cell_3/dropout_5/GreaterEqualGreaterEqualBlstm_2/lstm_cell_3/dropout_5/random_uniform/RandomUniform:output:04lstm_2/lstm_cell_3/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
!lstm_2/lstm_cell_3/dropout_5/CastCast-lstm_2/lstm_cell_3/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¨
"lstm_2/lstm_cell_3/dropout_5/Mul_1Mul$lstm_2/lstm_cell_3/dropout_5/Mul:z:0%lstm_2/lstm_cell_3/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
"lstm_2/lstm_cell_3/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¯
 lstm_2/lstm_cell_3/dropout_6/MulMul'lstm_2/lstm_cell_3/ones_like_1:output:0+lstm_2/lstm_cell_3/dropout_6/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
"lstm_2/lstm_cell_3/dropout_6/ShapeShape'lstm_2/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:¶
9lstm_2/lstm_cell_3/dropout_6/random_uniform/RandomUniformRandomUniform+lstm_2/lstm_cell_3/dropout_6/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0p
+lstm_2/lstm_cell_3/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)lstm_2/lstm_cell_3/dropout_6/GreaterEqualGreaterEqualBlstm_2/lstm_cell_3/dropout_6/random_uniform/RandomUniform:output:04lstm_2/lstm_cell_3/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
!lstm_2/lstm_cell_3/dropout_6/CastCast-lstm_2/lstm_cell_3/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¨
"lstm_2/lstm_cell_3/dropout_6/Mul_1Mul$lstm_2/lstm_cell_3/dropout_6/Mul:z:0%lstm_2/lstm_cell_3/dropout_6/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
"lstm_2/lstm_cell_3/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¯
 lstm_2/lstm_cell_3/dropout_7/MulMul'lstm_2/lstm_cell_3/ones_like_1:output:0+lstm_2/lstm_cell_3/dropout_7/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
"lstm_2/lstm_cell_3/dropout_7/ShapeShape'lstm_2/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:¶
9lstm_2/lstm_cell_3/dropout_7/random_uniform/RandomUniformRandomUniform+lstm_2/lstm_cell_3/dropout_7/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0p
+lstm_2/lstm_cell_3/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)lstm_2/lstm_cell_3/dropout_7/GreaterEqualGreaterEqualBlstm_2/lstm_cell_3/dropout_7/random_uniform/RandomUniform:output:04lstm_2/lstm_cell_3/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
!lstm_2/lstm_cell_3/dropout_7/CastCast-lstm_2/lstm_cell_3/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¨
"lstm_2/lstm_cell_3/dropout_7/Mul_1Mul$lstm_2/lstm_cell_3/dropout_7/Mul:z:0%lstm_2/lstm_cell_3/dropout_7/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/lstm_cell_3/mulMullstm_2/strided_slice_2:output:0$lstm_2/lstm_cell_3/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_2/lstm_cell_3/mul_1Mullstm_2/strided_slice_2:output:0&lstm_2/lstm_cell_3/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_2/lstm_cell_3/mul_2Mullstm_2/strided_slice_2:output:0&lstm_2/lstm_cell_3/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_2/lstm_cell_3/mul_3Mullstm_2/strided_slice_2:output:0&lstm_2/lstm_cell_3/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
"lstm_2/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
'lstm_2/lstm_cell_3/split/ReadVariableOpReadVariableOp0lstm_2_lstm_cell_3_split_readvariableop_resource*
_output_shapes
:	d*
dtype0×
lstm_2/lstm_cell_3/splitSplit+lstm_2/lstm_cell_3/split/split_dim:output:0/lstm_2/lstm_cell_3/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:d@:d@:d@:d@*
	num_split
lstm_2/lstm_cell_3/MatMulMatMullstm_2/lstm_cell_3/mul:z:0!lstm_2/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/lstm_cell_3/MatMul_1MatMullstm_2/lstm_cell_3/mul_1:z:0!lstm_2/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/lstm_cell_3/MatMul_2MatMullstm_2/lstm_cell_3/mul_2:z:0!lstm_2/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/lstm_cell_3/MatMul_3MatMullstm_2/lstm_cell_3/mul_3:z:0!lstm_2/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@f
$lstm_2/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
)lstm_2/lstm_cell_3/split_1/ReadVariableOpReadVariableOp2lstm_2_lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0Í
lstm_2/lstm_cell_3/split_1Split-lstm_2/lstm_cell_3/split_1/split_dim:output:01lstm_2/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split¡
lstm_2/lstm_cell_3/BiasAddBiasAdd#lstm_2/lstm_cell_3/MatMul:product:0#lstm_2/lstm_cell_3/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
lstm_2/lstm_cell_3/BiasAdd_1BiasAdd%lstm_2/lstm_cell_3/MatMul_1:product:0#lstm_2/lstm_cell_3/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
lstm_2/lstm_cell_3/BiasAdd_2BiasAdd%lstm_2/lstm_cell_3/MatMul_2:product:0#lstm_2/lstm_cell_3/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
lstm_2/lstm_cell_3/BiasAdd_3BiasAdd%lstm_2/lstm_cell_3/MatMul_3:product:0#lstm_2/lstm_cell_3/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/lstm_cell_3/mul_4Mullstm_2/zeros:output:0&lstm_2/lstm_cell_3/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/lstm_cell_3/mul_5Mullstm_2/zeros:output:0&lstm_2/lstm_cell_3/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/lstm_cell_3/mul_6Mullstm_2/zeros:output:0&lstm_2/lstm_cell_3/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/lstm_cell_3/mul_7Mullstm_2/zeros:output:0&lstm_2/lstm_cell_3/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
!lstm_2/lstm_cell_3/ReadVariableOpReadVariableOp*lstm_2_lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0w
&lstm_2/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(lstm_2/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   y
(lstm_2/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ê
 lstm_2/lstm_cell_3/strided_sliceStridedSlice)lstm_2/lstm_cell_3/ReadVariableOp:value:0/lstm_2/lstm_cell_3/strided_slice/stack:output:01lstm_2/lstm_cell_3/strided_slice/stack_1:output:01lstm_2/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask 
lstm_2/lstm_cell_3/MatMul_4MatMullstm_2/lstm_cell_3/mul_4:z:0)lstm_2/lstm_cell_3/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/lstm_cell_3/addAddV2#lstm_2/lstm_cell_3/BiasAdd:output:0%lstm_2/lstm_cell_3/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
lstm_2/lstm_cell_3/SigmoidSigmoidlstm_2/lstm_cell_3/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
#lstm_2/lstm_cell_3/ReadVariableOp_1ReadVariableOp*lstm_2_lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0y
(lstm_2/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   {
*lstm_2/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_2/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
"lstm_2/lstm_cell_3/strided_slice_1StridedSlice+lstm_2/lstm_cell_3/ReadVariableOp_1:value:01lstm_2/lstm_cell_3/strided_slice_1/stack:output:03lstm_2/lstm_cell_3/strided_slice_1/stack_1:output:03lstm_2/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask¢
lstm_2/lstm_cell_3/MatMul_5MatMullstm_2/lstm_cell_3/mul_5:z:0+lstm_2/lstm_cell_3/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡
lstm_2/lstm_cell_3/add_1AddV2%lstm_2/lstm_cell_3/BiasAdd_1:output:0%lstm_2/lstm_cell_3/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
lstm_2/lstm_cell_3/Sigmoid_1Sigmoidlstm_2/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/lstm_cell_3/mul_8Mul lstm_2/lstm_cell_3/Sigmoid_1:y:0lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
#lstm_2/lstm_cell_3/ReadVariableOp_2ReadVariableOp*lstm_2_lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0y
(lstm_2/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_2/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   {
*lstm_2/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
"lstm_2/lstm_cell_3/strided_slice_2StridedSlice+lstm_2/lstm_cell_3/ReadVariableOp_2:value:01lstm_2/lstm_cell_3/strided_slice_2/stack:output:03lstm_2/lstm_cell_3/strided_slice_2/stack_1:output:03lstm_2/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask¢
lstm_2/lstm_cell_3/MatMul_6MatMullstm_2/lstm_cell_3/mul_6:z:0+lstm_2/lstm_cell_3/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡
lstm_2/lstm_cell_3/add_2AddV2%lstm_2/lstm_cell_3/BiasAdd_2:output:0%lstm_2/lstm_cell_3/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
lstm_2/lstm_cell_3/TanhTanhlstm_2/lstm_cell_3/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/lstm_cell_3/mul_9Mullstm_2/lstm_cell_3/Sigmoid:y:0lstm_2/lstm_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/lstm_cell_3/add_3AddV2lstm_2/lstm_cell_3/mul_8:z:0lstm_2/lstm_cell_3/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
#lstm_2/lstm_cell_3/ReadVariableOp_3ReadVariableOp*lstm_2_lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0y
(lstm_2/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   {
*lstm_2/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_2/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
"lstm_2/lstm_cell_3/strided_slice_3StridedSlice+lstm_2/lstm_cell_3/ReadVariableOp_3:value:01lstm_2/lstm_cell_3/strided_slice_3/stack:output:03lstm_2/lstm_cell_3/strided_slice_3/stack_1:output:03lstm_2/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask¢
lstm_2/lstm_cell_3/MatMul_7MatMullstm_2/lstm_cell_3/mul_7:z:0+lstm_2/lstm_cell_3/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡
lstm_2/lstm_cell_3/add_4AddV2%lstm_2/lstm_cell_3/BiasAdd_3:output:0%lstm_2/lstm_cell_3/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
lstm_2/lstm_cell_3/Sigmoid_2Sigmoidlstm_2/lstm_cell_3/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
lstm_2/lstm_cell_3/Tanh_1Tanhlstm_2/lstm_cell_3/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/lstm_cell_3/mul_10Mul lstm_2/lstm_cell_3/Sigmoid_2:y:0lstm_2/lstm_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@u
$lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Í
lstm_2/TensorArrayV2_1TensorListReserve-lstm_2/TensorArrayV2_1/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒM
lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Õ
lstm_2/whileWhile"lstm_2/while/loop_counter:output:0(lstm_2/while/maximum_iterations:output:0lstm_2/time:output:0lstm_2/TensorArrayV2_1:handle:0lstm_2/zeros:output:0lstm_2/zeros_1:output:0lstm_2/strided_slice_1:output:0>lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:00lstm_2_lstm_cell_3_split_readvariableop_resource2lstm_2_lstm_cell_3_split_1_readvariableop_resource*lstm_2_lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstm_2_while_body_23634*#
condR
lstm_2_while_cond_23633*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
7lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ø
)lstm_2/TensorArrayV2Stack/TensorListStackTensorListStacklstm_2/while:output:3@lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:úÿÿÿÿÿÿÿÿÿ@*
element_dtype0o
lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿh
lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ª
lstm_2/strided_slice_3StridedSlice2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_2/strided_slice_3/stack:output:0'lstm_2/strided_slice_3/stack_1:output:0'lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maskl
lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¬
lstm_2/transpose_1	Transpose2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_2/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿú@b
lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
dense_4/MatMulMatMullstm_2/strided_slice_3:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitydense_5/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^embedding_4/embedding_lookup"^lstm_2/lstm_cell_3/ReadVariableOp$^lstm_2/lstm_cell_3/ReadVariableOp_1$^lstm_2/lstm_cell_3/ReadVariableOp_2$^lstm_2/lstm_cell_3/ReadVariableOp_3(^lstm_2/lstm_cell_3/split/ReadVariableOp*^lstm_2/lstm_cell_3/split_1/ReadVariableOp^lstm_2/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿú: : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2<
embedding_4/embedding_lookupembedding_4/embedding_lookup2F
!lstm_2/lstm_cell_3/ReadVariableOp!lstm_2/lstm_cell_3/ReadVariableOp2J
#lstm_2/lstm_cell_3/ReadVariableOp_1#lstm_2/lstm_cell_3/ReadVariableOp_12J
#lstm_2/lstm_cell_3/ReadVariableOp_2#lstm_2/lstm_cell_3/ReadVariableOp_22J
#lstm_2/lstm_cell_3/ReadVariableOp_3#lstm_2/lstm_cell_3/ReadVariableOp_32R
'lstm_2/lstm_cell_3/split/ReadVariableOp'lstm_2/lstm_cell_3/split/ReadVariableOp2V
)lstm_2/lstm_cell_3/split_1/ReadVariableOp)lstm_2/lstm_cell_3/split_1/ReadVariableOp2
lstm_2/whilelstm_2/while:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs
ßØ

lstm_2_while_body_23634*
&lstm_2_while_lstm_2_while_loop_counter0
,lstm_2_while_lstm_2_while_maximum_iterations
lstm_2_while_placeholder
lstm_2_while_placeholder_1
lstm_2_while_placeholder_2
lstm_2_while_placeholder_3)
%lstm_2_while_lstm_2_strided_slice_1_0e
alstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0K
8lstm_2_while_lstm_cell_3_split_readvariableop_resource_0:	dI
:lstm_2_while_lstm_cell_3_split_1_readvariableop_resource_0:	E
2lstm_2_while_lstm_cell_3_readvariableop_resource_0:	@
lstm_2_while_identity
lstm_2_while_identity_1
lstm_2_while_identity_2
lstm_2_while_identity_3
lstm_2_while_identity_4
lstm_2_while_identity_5'
#lstm_2_while_lstm_2_strided_slice_1c
_lstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensorI
6lstm_2_while_lstm_cell_3_split_readvariableop_resource:	dG
8lstm_2_while_lstm_cell_3_split_1_readvariableop_resource:	C
0lstm_2_while_lstm_cell_3_readvariableop_resource:	@¢'lstm_2/while/lstm_cell_3/ReadVariableOp¢)lstm_2/while/lstm_cell_3/ReadVariableOp_1¢)lstm_2/while/lstm_cell_3/ReadVariableOp_2¢)lstm_2/while/lstm_cell_3/ReadVariableOp_3¢-lstm_2/while/lstm_cell_3/split/ReadVariableOp¢/lstm_2/while/lstm_cell_3/split_1/ReadVariableOp
>lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   É
0lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0lstm_2_while_placeholderGlstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
(lstm_2/while/lstm_cell_3/ones_like/ShapeShape7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:m
(lstm_2/while/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Â
"lstm_2/while/lstm_cell_3/ones_likeFill1lstm_2/while/lstm_cell_3/ones_like/Shape:output:01lstm_2/while/lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
&lstm_2/while/lstm_cell_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?»
$lstm_2/while/lstm_cell_3/dropout/MulMul+lstm_2/while/lstm_cell_3/ones_like:output:0/lstm_2/while/lstm_cell_3/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
&lstm_2/while/lstm_cell_3/dropout/ShapeShape+lstm_2/while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:¾
=lstm_2/while/lstm_cell_3/dropout/random_uniform/RandomUniformRandomUniform/lstm_2/while/lstm_cell_3/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0t
/lstm_2/while/lstm_cell_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ñ
-lstm_2/while/lstm_cell_3/dropout/GreaterEqualGreaterEqualFlstm_2/while/lstm_cell_3/dropout/random_uniform/RandomUniform:output:08lstm_2/while/lstm_cell_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¡
%lstm_2/while/lstm_cell_3/dropout/CastCast1lstm_2/while/lstm_cell_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd´
&lstm_2/while/lstm_cell_3/dropout/Mul_1Mul(lstm_2/while/lstm_cell_3/dropout/Mul:z:0)lstm_2/while/lstm_cell_3/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdm
(lstm_2/while/lstm_cell_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¿
&lstm_2/while/lstm_cell_3/dropout_1/MulMul+lstm_2/while/lstm_cell_3/ones_like:output:01lstm_2/while/lstm_cell_3/dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(lstm_2/while/lstm_cell_3/dropout_1/ShapeShape+lstm_2/while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:Â
?lstm_2/while/lstm_cell_3/dropout_1/random_uniform/RandomUniformRandomUniform1lstm_2/while/lstm_cell_3/dropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0v
1lstm_2/while/lstm_cell_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=÷
/lstm_2/while/lstm_cell_3/dropout_1/GreaterEqualGreaterEqualHlstm_2/while/lstm_cell_3/dropout_1/random_uniform/RandomUniform:output:0:lstm_2/while/lstm_cell_3/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¥
'lstm_2/while/lstm_cell_3/dropout_1/CastCast3lstm_2/while/lstm_cell_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdº
(lstm_2/while/lstm_cell_3/dropout_1/Mul_1Mul*lstm_2/while/lstm_cell_3/dropout_1/Mul:z:0+lstm_2/while/lstm_cell_3/dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdm
(lstm_2/while/lstm_cell_3/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¿
&lstm_2/while/lstm_cell_3/dropout_2/MulMul+lstm_2/while/lstm_cell_3/ones_like:output:01lstm_2/while/lstm_cell_3/dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(lstm_2/while/lstm_cell_3/dropout_2/ShapeShape+lstm_2/while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:Â
?lstm_2/while/lstm_cell_3/dropout_2/random_uniform/RandomUniformRandomUniform1lstm_2/while/lstm_cell_3/dropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0v
1lstm_2/while/lstm_cell_3/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=÷
/lstm_2/while/lstm_cell_3/dropout_2/GreaterEqualGreaterEqualHlstm_2/while/lstm_cell_3/dropout_2/random_uniform/RandomUniform:output:0:lstm_2/while/lstm_cell_3/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¥
'lstm_2/while/lstm_cell_3/dropout_2/CastCast3lstm_2/while/lstm_cell_3/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdº
(lstm_2/while/lstm_cell_3/dropout_2/Mul_1Mul*lstm_2/while/lstm_cell_3/dropout_2/Mul:z:0+lstm_2/while/lstm_cell_3/dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdm
(lstm_2/while/lstm_cell_3/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¿
&lstm_2/while/lstm_cell_3/dropout_3/MulMul+lstm_2/while/lstm_cell_3/ones_like:output:01lstm_2/while/lstm_cell_3/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(lstm_2/while/lstm_cell_3/dropout_3/ShapeShape+lstm_2/while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:Â
?lstm_2/while/lstm_cell_3/dropout_3/random_uniform/RandomUniformRandomUniform1lstm_2/while/lstm_cell_3/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0v
1lstm_2/while/lstm_cell_3/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=÷
/lstm_2/while/lstm_cell_3/dropout_3/GreaterEqualGreaterEqualHlstm_2/while/lstm_cell_3/dropout_3/random_uniform/RandomUniform:output:0:lstm_2/while/lstm_cell_3/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¥
'lstm_2/while/lstm_cell_3/dropout_3/CastCast3lstm_2/while/lstm_cell_3/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdº
(lstm_2/while/lstm_cell_3/dropout_3/Mul_1Mul*lstm_2/while/lstm_cell_3/dropout_3/Mul:z:0+lstm_2/while/lstm_cell_3/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
*lstm_2/while/lstm_cell_3/ones_like_1/ShapeShapelstm_2_while_placeholder_2*
T0*
_output_shapes
:o
*lstm_2/while/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?È
$lstm_2/while/lstm_cell_3/ones_like_1Fill3lstm_2/while/lstm_cell_3/ones_like_1/Shape:output:03lstm_2/while/lstm_cell_3/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
(lstm_2/while/lstm_cell_3/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?Á
&lstm_2/while/lstm_cell_3/dropout_4/MulMul-lstm_2/while/lstm_cell_3/ones_like_1:output:01lstm_2/while/lstm_cell_3/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(lstm_2/while/lstm_cell_3/dropout_4/ShapeShape-lstm_2/while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:Â
?lstm_2/while/lstm_cell_3/dropout_4/random_uniform/RandomUniformRandomUniform1lstm_2/while/lstm_cell_3/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0v
1lstm_2/while/lstm_cell_3/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=÷
/lstm_2/while/lstm_cell_3/dropout_4/GreaterEqualGreaterEqualHlstm_2/while/lstm_cell_3/dropout_4/random_uniform/RandomUniform:output:0:lstm_2/while/lstm_cell_3/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
'lstm_2/while/lstm_cell_3/dropout_4/CastCast3lstm_2/while/lstm_cell_3/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
(lstm_2/while/lstm_cell_3/dropout_4/Mul_1Mul*lstm_2/while/lstm_cell_3/dropout_4/Mul:z:0+lstm_2/while/lstm_cell_3/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
(lstm_2/while/lstm_cell_3/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?Á
&lstm_2/while/lstm_cell_3/dropout_5/MulMul-lstm_2/while/lstm_cell_3/ones_like_1:output:01lstm_2/while/lstm_cell_3/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(lstm_2/while/lstm_cell_3/dropout_5/ShapeShape-lstm_2/while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:Â
?lstm_2/while/lstm_cell_3/dropout_5/random_uniform/RandomUniformRandomUniform1lstm_2/while/lstm_cell_3/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0v
1lstm_2/while/lstm_cell_3/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=÷
/lstm_2/while/lstm_cell_3/dropout_5/GreaterEqualGreaterEqualHlstm_2/while/lstm_cell_3/dropout_5/random_uniform/RandomUniform:output:0:lstm_2/while/lstm_cell_3/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
'lstm_2/while/lstm_cell_3/dropout_5/CastCast3lstm_2/while/lstm_cell_3/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
(lstm_2/while/lstm_cell_3/dropout_5/Mul_1Mul*lstm_2/while/lstm_cell_3/dropout_5/Mul:z:0+lstm_2/while/lstm_cell_3/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
(lstm_2/while/lstm_cell_3/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?Á
&lstm_2/while/lstm_cell_3/dropout_6/MulMul-lstm_2/while/lstm_cell_3/ones_like_1:output:01lstm_2/while/lstm_cell_3/dropout_6/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(lstm_2/while/lstm_cell_3/dropout_6/ShapeShape-lstm_2/while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:Â
?lstm_2/while/lstm_cell_3/dropout_6/random_uniform/RandomUniformRandomUniform1lstm_2/while/lstm_cell_3/dropout_6/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0v
1lstm_2/while/lstm_cell_3/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=÷
/lstm_2/while/lstm_cell_3/dropout_6/GreaterEqualGreaterEqualHlstm_2/while/lstm_cell_3/dropout_6/random_uniform/RandomUniform:output:0:lstm_2/while/lstm_cell_3/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
'lstm_2/while/lstm_cell_3/dropout_6/CastCast3lstm_2/while/lstm_cell_3/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
(lstm_2/while/lstm_cell_3/dropout_6/Mul_1Mul*lstm_2/while/lstm_cell_3/dropout_6/Mul:z:0+lstm_2/while/lstm_cell_3/dropout_6/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
(lstm_2/while/lstm_cell_3/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?Á
&lstm_2/while/lstm_cell_3/dropout_7/MulMul-lstm_2/while/lstm_cell_3/ones_like_1:output:01lstm_2/while/lstm_cell_3/dropout_7/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(lstm_2/while/lstm_cell_3/dropout_7/ShapeShape-lstm_2/while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:Â
?lstm_2/while/lstm_cell_3/dropout_7/random_uniform/RandomUniformRandomUniform1lstm_2/while/lstm_cell_3/dropout_7/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0v
1lstm_2/while/lstm_cell_3/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=÷
/lstm_2/while/lstm_cell_3/dropout_7/GreaterEqualGreaterEqualHlstm_2/while/lstm_cell_3/dropout_7/random_uniform/RandomUniform:output:0:lstm_2/while/lstm_cell_3/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
'lstm_2/while/lstm_cell_3/dropout_7/CastCast3lstm_2/while/lstm_cell_3/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
(lstm_2/while/lstm_cell_3/dropout_7/Mul_1Mul*lstm_2/while/lstm_cell_3/dropout_7/Mul:z:0+lstm_2/while/lstm_cell_3/dropout_7/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
lstm_2/while/lstm_cell_3/mulMul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0*lstm_2/while/lstm_cell_3/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¾
lstm_2/while/lstm_cell_3/mul_1Mul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0,lstm_2/while/lstm_cell_3/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¾
lstm_2/while/lstm_cell_3/mul_2Mul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0,lstm_2/while/lstm_cell_3/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¾
lstm_2/while/lstm_cell_3/mul_3Mul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0,lstm_2/while/lstm_cell_3/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdj
(lstm_2/while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :§
-lstm_2/while/lstm_cell_3/split/ReadVariableOpReadVariableOp8lstm_2_while_lstm_cell_3_split_readvariableop_resource_0*
_output_shapes
:	d*
dtype0é
lstm_2/while/lstm_cell_3/splitSplit1lstm_2/while/lstm_cell_3/split/split_dim:output:05lstm_2/while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:d@:d@:d@:d@*
	num_split¦
lstm_2/while/lstm_cell_3/MatMulMatMul lstm_2/while/lstm_cell_3/mul:z:0'lstm_2/while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ª
!lstm_2/while/lstm_cell_3/MatMul_1MatMul"lstm_2/while/lstm_cell_3/mul_1:z:0'lstm_2/while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ª
!lstm_2/while/lstm_cell_3/MatMul_2MatMul"lstm_2/while/lstm_cell_3/mul_2:z:0'lstm_2/while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ª
!lstm_2/while/lstm_cell_3/MatMul_3MatMul"lstm_2/while/lstm_cell_3/mul_3:z:0'lstm_2/while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
*lstm_2/while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : §
/lstm_2/while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp:lstm_2_while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0ß
 lstm_2/while/lstm_cell_3/split_1Split3lstm_2/while/lstm_cell_3/split_1/split_dim:output:07lstm_2/while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split³
 lstm_2/while/lstm_cell_3/BiasAddBiasAdd)lstm_2/while/lstm_cell_3/MatMul:product:0)lstm_2/while/lstm_cell_3/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@·
"lstm_2/while/lstm_cell_3/BiasAdd_1BiasAdd+lstm_2/while/lstm_cell_3/MatMul_1:product:0)lstm_2/while/lstm_cell_3/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@·
"lstm_2/while/lstm_cell_3/BiasAdd_2BiasAdd+lstm_2/while/lstm_cell_3/MatMul_2:product:0)lstm_2/while/lstm_cell_3/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@·
"lstm_2/while/lstm_cell_3/BiasAdd_3BiasAdd+lstm_2/while/lstm_cell_3/MatMul_3:product:0)lstm_2/while/lstm_cell_3/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡
lstm_2/while/lstm_cell_3/mul_4Mullstm_2_while_placeholder_2,lstm_2/while/lstm_cell_3/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡
lstm_2/while/lstm_cell_3/mul_5Mullstm_2_while_placeholder_2,lstm_2/while/lstm_cell_3/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡
lstm_2/while/lstm_cell_3/mul_6Mullstm_2_while_placeholder_2,lstm_2/while/lstm_cell_3/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡
lstm_2/while/lstm_cell_3/mul_7Mullstm_2_while_placeholder_2,lstm_2/while/lstm_cell_3/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
'lstm_2/while/lstm_cell_3/ReadVariableOpReadVariableOp2lstm_2_while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0}
,lstm_2/while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.lstm_2/while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   
.lstm_2/while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      è
&lstm_2/while/lstm_cell_3/strided_sliceStridedSlice/lstm_2/while/lstm_cell_3/ReadVariableOp:value:05lstm_2/while/lstm_cell_3/strided_slice/stack:output:07lstm_2/while/lstm_cell_3/strided_slice/stack_1:output:07lstm_2/while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask²
!lstm_2/while/lstm_cell_3/MatMul_4MatMul"lstm_2/while/lstm_cell_3/mul_4:z:0/lstm_2/while/lstm_cell_3/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
lstm_2/while/lstm_cell_3/addAddV2)lstm_2/while/lstm_cell_3/BiasAdd:output:0+lstm_2/while/lstm_cell_3/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 lstm_2/while/lstm_cell_3/SigmoidSigmoid lstm_2/while/lstm_cell_3/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)lstm_2/while/lstm_cell_3/ReadVariableOp_1ReadVariableOp2lstm_2_while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0
.lstm_2/while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   
0lstm_2/while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
0lstm_2/while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ò
(lstm_2/while/lstm_cell_3/strided_slice_1StridedSlice1lstm_2/while/lstm_cell_3/ReadVariableOp_1:value:07lstm_2/while/lstm_cell_3/strided_slice_1/stack:output:09lstm_2/while/lstm_cell_3/strided_slice_1/stack_1:output:09lstm_2/while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask´
!lstm_2/while/lstm_cell_3/MatMul_5MatMul"lstm_2/while/lstm_cell_3/mul_5:z:01lstm_2/while/lstm_cell_3/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@³
lstm_2/while/lstm_cell_3/add_1AddV2+lstm_2/while/lstm_cell_3/BiasAdd_1:output:0+lstm_2/while/lstm_cell_3/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"lstm_2/while/lstm_cell_3/Sigmoid_1Sigmoid"lstm_2/while/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/while/lstm_cell_3/mul_8Mul&lstm_2/while/lstm_cell_3/Sigmoid_1:y:0lstm_2_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)lstm_2/while/lstm_cell_3/ReadVariableOp_2ReadVariableOp2lstm_2_while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0
.lstm_2/while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
0lstm_2/while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   
0lstm_2/while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ò
(lstm_2/while/lstm_cell_3/strided_slice_2StridedSlice1lstm_2/while/lstm_cell_3/ReadVariableOp_2:value:07lstm_2/while/lstm_cell_3/strided_slice_2/stack:output:09lstm_2/while/lstm_cell_3/strided_slice_2/stack_1:output:09lstm_2/while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask´
!lstm_2/while/lstm_cell_3/MatMul_6MatMul"lstm_2/while/lstm_cell_3/mul_6:z:01lstm_2/while/lstm_cell_3/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@³
lstm_2/while/lstm_cell_3/add_2AddV2+lstm_2/while/lstm_cell_3/BiasAdd_2:output:0+lstm_2/while/lstm_cell_3/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
lstm_2/while/lstm_cell_3/TanhTanh"lstm_2/while/lstm_cell_3/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
lstm_2/while/lstm_cell_3/mul_9Mul$lstm_2/while/lstm_cell_3/Sigmoid:y:0!lstm_2/while/lstm_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡
lstm_2/while/lstm_cell_3/add_3AddV2"lstm_2/while/lstm_cell_3/mul_8:z:0"lstm_2/while/lstm_cell_3/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)lstm_2/while/lstm_cell_3/ReadVariableOp_3ReadVariableOp2lstm_2_while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0
.lstm_2/while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   
0lstm_2/while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
0lstm_2/while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ò
(lstm_2/while/lstm_cell_3/strided_slice_3StridedSlice1lstm_2/while/lstm_cell_3/ReadVariableOp_3:value:07lstm_2/while/lstm_cell_3/strided_slice_3/stack:output:09lstm_2/while/lstm_cell_3/strided_slice_3/stack_1:output:09lstm_2/while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask´
!lstm_2/while/lstm_cell_3/MatMul_7MatMul"lstm_2/while/lstm_cell_3/mul_7:z:01lstm_2/while/lstm_cell_3/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@³
lstm_2/while/lstm_cell_3/add_4AddV2+lstm_2/while/lstm_cell_3/BiasAdd_3:output:0+lstm_2/while/lstm_cell_3/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"lstm_2/while/lstm_cell_3/Sigmoid_2Sigmoid"lstm_2/while/lstm_cell_3/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@}
lstm_2/while/lstm_cell_3/Tanh_1Tanh"lstm_2/while/lstm_cell_3/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
lstm_2/while/lstm_cell_3/mul_10Mul&lstm_2/while/lstm_cell_3/Sigmoid_2:y:0#lstm_2/while/lstm_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@á
1lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_2_while_placeholder_1lstm_2_while_placeholder#lstm_2/while/lstm_cell_3/mul_10:z:0*
_output_shapes
: *
element_dtype0:éèÒT
lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_2/while/addAddV2lstm_2_while_placeholderlstm_2/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_2/while/add_1AddV2&lstm_2_while_lstm_2_while_loop_counterlstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_2/while/IdentityIdentitylstm_2/while/add_1:z:0^lstm_2/while/NoOp*
T0*
_output_shapes
: 
lstm_2/while/Identity_1Identity,lstm_2_while_lstm_2_while_maximum_iterations^lstm_2/while/NoOp*
T0*
_output_shapes
: n
lstm_2/while/Identity_2Identitylstm_2/while/add:z:0^lstm_2/while/NoOp*
T0*
_output_shapes
: ®
lstm_2/while/Identity_3IdentityAlstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_2/while/NoOp*
T0*
_output_shapes
: :éèÒ
lstm_2/while/Identity_4Identity#lstm_2/while/lstm_cell_3/mul_10:z:0^lstm_2/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/while/Identity_5Identity"lstm_2/while/lstm_cell_3/add_3:z:0^lstm_2/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ã
lstm_2/while/NoOpNoOp(^lstm_2/while/lstm_cell_3/ReadVariableOp*^lstm_2/while/lstm_cell_3/ReadVariableOp_1*^lstm_2/while/lstm_cell_3/ReadVariableOp_2*^lstm_2/while/lstm_cell_3/ReadVariableOp_3.^lstm_2/while/lstm_cell_3/split/ReadVariableOp0^lstm_2/while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_2_while_identitylstm_2/while/Identity:output:0";
lstm_2_while_identity_1 lstm_2/while/Identity_1:output:0";
lstm_2_while_identity_2 lstm_2/while/Identity_2:output:0";
lstm_2_while_identity_3 lstm_2/while/Identity_3:output:0";
lstm_2_while_identity_4 lstm_2/while/Identity_4:output:0";
lstm_2_while_identity_5 lstm_2/while/Identity_5:output:0"L
#lstm_2_while_lstm_2_strided_slice_1%lstm_2_while_lstm_2_strided_slice_1_0"f
0lstm_2_while_lstm_cell_3_readvariableop_resource2lstm_2_while_lstm_cell_3_readvariableop_resource_0"v
8lstm_2_while_lstm_cell_3_split_1_readvariableop_resource:lstm_2_while_lstm_cell_3_split_1_readvariableop_resource_0"r
6lstm_2_while_lstm_cell_3_split_readvariableop_resource8lstm_2_while_lstm_cell_3_split_readvariableop_resource_0"Ä
_lstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensoralstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : 2R
'lstm_2/while/lstm_cell_3/ReadVariableOp'lstm_2/while/lstm_cell_3/ReadVariableOp2V
)lstm_2/while/lstm_cell_3/ReadVariableOp_1)lstm_2/while/lstm_cell_3/ReadVariableOp_12V
)lstm_2/while/lstm_cell_3/ReadVariableOp_2)lstm_2/while/lstm_cell_3/ReadVariableOp_22V
)lstm_2/while/lstm_cell_3/ReadVariableOp_3)lstm_2/while/lstm_cell_3/ReadVariableOp_32^
-lstm_2/while/lstm_cell_3/split/ReadVariableOp-lstm_2/while/lstm_cell_3/split/ReadVariableOp2b
/lstm_2/while/lstm_cell_3/split_1/ReadVariableOp/lstm_2/while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
Ñ
ò
G__inference_sequential_4_layer_call_and_return_conditional_losses_23056

inputs$
embedding_4_23035:	Äd
lstm_2_23038:	d
lstm_2_23040:	
lstm_2_23042:	@
dense_4_23045:@ 
dense_4_23047: 
dense_5_23050: 
dense_5_23052:
identity¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢#embedding_4/StatefulPartitionedCall¢lstm_2/StatefulPartitionedCallé
#embedding_4/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_4_23035*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúd*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_embedding_4_layer_call_and_return_conditional_losses_22280
lstm_2/StatefulPartitionedCallStatefulPartitionedCall,embedding_4/StatefulPartitionedCall:output:0lstm_2_23038lstm_2_23040lstm_2_23042*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_22992
dense_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0dense_4_23045dense_4_23047*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_22545
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_23050dense_5_23052*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_22562w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
NoOpNoOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall$^embedding_4/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿú: : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2J
#embedding_4/StatefulPartitionedCall#embedding_4/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs

µ
&__inference_lstm_2_layer_call_fn_23897
inputs_0
unknown:	d
	unknown_0:	
	unknown_1:	@
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_21949o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
inputs/0
Ð"
Õ
while_body_21880
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_3_21904_0:	d(
while_lstm_cell_3_21906_0:	,
while_lstm_cell_3_21908_0:	@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_3_21904:	d&
while_lstm_cell_3_21906:	*
while_lstm_cell_3_21908:	@¢)while/lstm_cell_3/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0ª
)while/lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_3_21904_0while_lstm_cell_3_21906_0while_lstm_cell_3_21908_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_21866Û
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_3/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity2while/lstm_cell_3/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/Identity_5Identity2while/lstm_cell_3/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x

while/NoOpNoOp*^while/lstm_cell_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_3_21904while_lstm_cell_3_21904_0"4
while_lstm_cell_3_21906while_lstm_cell_3_21906_0"4
while_lstm_cell_3_21908while_lstm_cell_3_21908_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : 2V
)while/lstm_cell_3/StatefulPartitionedCall)while/lstm_cell_3/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ò
ý
G__inference_sequential_4_layer_call_and_return_conditional_losses_23120
embedding_4_input$
embedding_4_23099:	Äd
lstm_2_23102:	d
lstm_2_23104:	
lstm_2_23106:	@
dense_4_23109:@ 
dense_4_23111: 
dense_5_23114: 
dense_5_23116:
identity¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢#embedding_4/StatefulPartitionedCall¢lstm_2/StatefulPartitionedCallô
#embedding_4/StatefulPartitionedCallStatefulPartitionedCallembedding_4_inputembedding_4_23099*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúd*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_embedding_4_layer_call_and_return_conditional_losses_22280
lstm_2/StatefulPartitionedCallStatefulPartitionedCall,embedding_4/StatefulPartitionedCall:output:0lstm_2_23102lstm_2_23104lstm_2_23106*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_22526
dense_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0dense_4_23109dense_4_23111*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_22545
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_23114dense_5_23116*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_22562w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
NoOpNoOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall$^embedding_4/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿú: : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2J
#embedding_4/StatefulPartitionedCall#embedding_4/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall:[ W
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
+
_user_specified_nameembedding_4_input
°
¾
while_cond_24652
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_24652___redundant_placeholder03
/while_while_cond_24652___redundant_placeholder13
/while_while_cond_24652___redundant_placeholder23
/while_while_cond_24652___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:

µ
&__inference_lstm_2_layer_call_fn_23908
inputs_0
unknown:	d
	unknown_0:	
	unknown_1:	@
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_22254o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
inputs/0
°
¾
while_cond_22184
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_22184___redundant_placeholder03
/while_while_cond_22184___redundant_placeholder13
/while_while_cond_22184___redundant_placeholder23
/while_while_cond_22184___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ï	
Î
,__inference_sequential_4_layer_call_fn_22588
embedding_4_input
unknown:	Äd
	unknown_0:	d
	unknown_1:	
	unknown_2:	@
	unknown_3:@ 
	unknown_4: 
	unknown_5: 
	unknown_6:
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallembedding_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_22569o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿú: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
+
_user_specified_nameembedding_4_input
®~
§
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_22126

inputs

states
states_10
split_readvariableop_resource:	d.
split_1_readvariableop_resource:	*
readvariableop_resource:	@
identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?p
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdO
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?t
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?t
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?t
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdG
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?v
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@S
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0]
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?v
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@S
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0]
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?v
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@S
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0]
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?v
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@S
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0]
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	d*
dtype0
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:d@:d@:d@:d@*
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[
mul_4Mulstatesdropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[
mul_5Mulstatesdropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[
mul_6Mulstatesdropout_6/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[
mul_7Mulstatesdropout_7/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ë
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskg
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      õ
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      õ
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@V
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      õ
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@K
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@À
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_namestates
Ð"
Õ
while_body_22185
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_3_22209_0:	d(
while_lstm_cell_3_22211_0:	,
while_lstm_cell_3_22213_0:	@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_3_22209:	d&
while_lstm_cell_3_22211:	*
while_lstm_cell_3_22213:	@¢)while/lstm_cell_3/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0ª
)while/lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_3_22209_0while_lstm_cell_3_22211_0while_lstm_cell_3_22213_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_22126Û
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_3/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity2while/lstm_cell_3/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/Identity_5Identity2while/lstm_cell_3/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x

while/NoOpNoOp*^while/lstm_cell_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_3_22209while_lstm_cell_3_22209_0"4
while_lstm_cell_3_22211while_lstm_cell_3_22211_0"4
while_lstm_cell_3_22213while_lstm_cell_3_22213_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : 2V
)while/lstm_cell_3/StatefulPartitionedCall)while/lstm_cell_3/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
¿t
	
while_body_22392
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_3_split_readvariableop_resource_0:	dB
3while_lstm_cell_3_split_1_readvariableop_resource_0:	>
+while_lstm_cell_3_readvariableop_resource_0:	@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_3_split_readvariableop_resource:	d@
1while_lstm_cell_3_split_1_readvariableop_resource:	<
)while_lstm_cell_3_readvariableop_resource:	@¢ while/lstm_cell_3/ReadVariableOp¢"while/lstm_cell_3/ReadVariableOp_1¢"while/lstm_cell_3/ReadVariableOp_2¢"while/lstm_cell_3/ReadVariableOp_3¢&while/lstm_cell_3/split/ReadVariableOp¢(while/lstm_cell_3/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
!while/lstm_cell_3/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?­
while/lstm_cell_3/ones_likeFill*while/lstm_cell_3/ones_like/Shape:output:0*while/lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
#while/lstm_cell_3/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:h
#while/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
while/lstm_cell_3/ones_like_1Fill,while/lstm_cell_3/ones_like_1/Shape:output:0,while/lstm_cell_3/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
while/lstm_cell_3/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¨
while/lstm_cell_3/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¨
while/lstm_cell_3/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¨
while/lstm_cell_3/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
&while/lstm_cell_3/split/ReadVariableOpReadVariableOp1while_lstm_cell_3_split_readvariableop_resource_0*
_output_shapes
:	d*
dtype0Ô
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0.while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:d@:d@:d@:d@*
	num_split
while/lstm_cell_3/MatMulMatMulwhile/lstm_cell_3/mul:z:0 while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/MatMul_1MatMulwhile/lstm_cell_3/mul_1:z:0 while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/MatMul_2MatMulwhile/lstm_cell_3/mul_2:z:0 while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/MatMul_3MatMulwhile/lstm_cell_3/mul_3:z:0 while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
#while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
(while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Ê
while/lstm_cell_3/split_1Split,while/lstm_cell_3/split_1/split_dim:output:00while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split
while/lstm_cell_3/BiasAddBiasAdd"while/lstm_cell_3/MatMul:product:0"while/lstm_cell_3/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
while/lstm_cell_3/BiasAdd_1BiasAdd$while/lstm_cell_3/MatMul_1:product:0"while/lstm_cell_3/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
while/lstm_cell_3/BiasAdd_2BiasAdd$while/lstm_cell_3/MatMul_2:product:0"while/lstm_cell_3/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
while/lstm_cell_3/BiasAdd_3BiasAdd$while/lstm_cell_3/MatMul_3:product:0"while/lstm_cell_3/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_4Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_5Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_6Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_7Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 while/lstm_cell_3/ReadVariableOpReadVariableOp+while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0v
%while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   x
'while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Å
while/lstm_cell_3/strided_sliceStridedSlice(while/lstm_cell_3/ReadVariableOp:value:0.while/lstm_cell_3/strided_slice/stack:output:00while/lstm_cell_3/strided_slice/stack_1:output:00while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_4MatMulwhile/lstm_cell_3/mul_4:z:0(while/lstm_cell_3/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/addAddV2"while/lstm_cell_3/BiasAdd:output:0$while/lstm_cell_3/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
while/lstm_cell_3/SigmoidSigmoidwhile/lstm_cell_3/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"while/lstm_cell_3/ReadVariableOp_1ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0x
'while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   z
)while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ï
!while/lstm_cell_3/strided_slice_1StridedSlice*while/lstm_cell_3/ReadVariableOp_1:value:00while/lstm_cell_3/strided_slice_1/stack:output:02while/lstm_cell_3/strided_slice_1/stack_1:output:02while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_5MatMulwhile/lstm_cell_3/mul_5:z:0*while/lstm_cell_3/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/add_1AddV2$while/lstm_cell_3/BiasAdd_1:output:0$while/lstm_cell_3/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@u
while/lstm_cell_3/Sigmoid_1Sigmoidwhile/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_8Mulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"while/lstm_cell_3/ReadVariableOp_2ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0x
'while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   z
)while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ï
!while/lstm_cell_3/strided_slice_2StridedSlice*while/lstm_cell_3/ReadVariableOp_2:value:00while/lstm_cell_3/strided_slice_2/stack:output:02while/lstm_cell_3/strided_slice_2/stack_1:output:02while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_6MatMulwhile/lstm_cell_3/mul_6:z:0*while/lstm_cell_3/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/add_2AddV2$while/lstm_cell_3/BiasAdd_2:output:0$while/lstm_cell_3/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
while/lstm_cell_3/TanhTanhwhile/lstm_cell_3/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_9Mulwhile/lstm_cell_3/Sigmoid:y:0while/lstm_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/add_3AddV2while/lstm_cell_3/mul_8:z:0while/lstm_cell_3/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"while/lstm_cell_3/ReadVariableOp_3ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0x
'while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   z
)while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ï
!while/lstm_cell_3/strided_slice_3StridedSlice*while/lstm_cell_3/ReadVariableOp_3:value:00while/lstm_cell_3/strided_slice_3/stack:output:02while/lstm_cell_3/strided_slice_3/stack_1:output:02while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_7MatMulwhile/lstm_cell_3/mul_7:z:0*while/lstm_cell_3/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/add_4AddV2$while/lstm_cell_3/BiasAdd_3:output:0$while/lstm_cell_3/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@u
while/lstm_cell_3/Sigmoid_2Sigmoidwhile/lstm_cell_3/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
while/lstm_cell_3/Tanh_1Tanhwhile/lstm_cell_3/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_10Mulwhile/lstm_cell_3/Sigmoid_2:y:0while/lstm_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_10:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒy
while/Identity_4Identitywhile/lstm_cell_3/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
while/Identity_5Identitywhile/lstm_cell_3/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@²

while/NoOpNoOp!^while/lstm_cell_3/ReadVariableOp#^while/lstm_cell_3/ReadVariableOp_1#^while/lstm_cell_3/ReadVariableOp_2#^while/lstm_cell_3/ReadVariableOp_3'^while/lstm_cell_3/split/ReadVariableOp)^while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_3_readvariableop_resource+while_lstm_cell_3_readvariableop_resource_0"h
1while_lstm_cell_3_split_1_readvariableop_resource3while_lstm_cell_3_split_1_readvariableop_resource_0"d
/while_lstm_cell_3_split_readvariableop_resource1while_lstm_cell_3_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : 2D
 while/lstm_cell_3/ReadVariableOp while/lstm_cell_3/ReadVariableOp2H
"while/lstm_cell_3/ReadVariableOp_1"while/lstm_cell_3/ReadVariableOp_12H
"while/lstm_cell_3/ReadVariableOp_2"while/lstm_cell_3/ReadVariableOp_22H
"while/lstm_cell_3/ReadVariableOp_3"while/lstm_cell_3/ReadVariableOp_32P
&while/lstm_cell_3/split/ReadVariableOp&while/lstm_cell_3/split/ReadVariableOp2T
(while/lstm_cell_3/split_1/ReadVariableOp(while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
õ
³
&__inference_lstm_2_layer_call_fn_23930

inputs
unknown:	d
	unknown_0:	
	unknown_1:	@
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_22992o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿúd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúd
 
_user_specified_nameinputs
}
°
!__inference__traced_restore_25679
file_prefix:
'assignvariableop_embedding_4_embeddings:	Äd3
!assignvariableop_1_dense_4_kernel:@ -
assignvariableop_2_dense_4_bias: 3
!assignvariableop_3_dense_5_kernel: -
assignvariableop_4_dense_5_bias:&
assignvariableop_5_adam_iter:	 (
assignvariableop_6_adam_beta_1: (
assignvariableop_7_adam_beta_2: '
assignvariableop_8_adam_decay: /
%assignvariableop_9_adam_learning_rate: @
-assignvariableop_10_lstm_2_lstm_cell_3_kernel:	dJ
7assignvariableop_11_lstm_2_lstm_cell_3_recurrent_kernel:	@:
+assignvariableop_12_lstm_2_lstm_cell_3_bias:	#
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: ;
)assignvariableop_17_adam_dense_4_kernel_m:@ 5
'assignvariableop_18_adam_dense_4_bias_m: ;
)assignvariableop_19_adam_dense_5_kernel_m: 5
'assignvariableop_20_adam_dense_5_bias_m:G
4assignvariableop_21_adam_lstm_2_lstm_cell_3_kernel_m:	dQ
>assignvariableop_22_adam_lstm_2_lstm_cell_3_recurrent_kernel_m:	@A
2assignvariableop_23_adam_lstm_2_lstm_cell_3_bias_m:	;
)assignvariableop_24_adam_dense_4_kernel_v:@ 5
'assignvariableop_25_adam_dense_4_bias_v: ;
)assignvariableop_26_adam_dense_5_kernel_v: 5
'assignvariableop_27_adam_dense_5_bias_v:G
4assignvariableop_28_adam_lstm_2_lstm_cell_3_kernel_v:	dQ
>assignvariableop_29_adam_lstm_2_lstm_cell_3_recurrent_kernel_v:	@A
2assignvariableop_30_adam_lstm_2_lstm_cell_3_bias_v:	
identity_32¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*°
value¦B£ B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH°
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Á
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::*.
dtypes$
"2 	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp'assignvariableop_embedding_4_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_4_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_4_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_5_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_5_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp-assignvariableop_10_lstm_2_lstm_cell_3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_11AssignVariableOp7assignvariableop_11_lstm_2_lstm_cell_3_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp+assignvariableop_12_lstm_2_lstm_cell_3_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_4_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_4_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_5_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_5_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_21AssignVariableOp4assignvariableop_21_adam_lstm_2_lstm_cell_3_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_22AssignVariableOp>assignvariableop_22_adam_lstm_2_lstm_cell_3_recurrent_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_23AssignVariableOp2assignvariableop_23_adam_lstm_2_lstm_cell_3_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_4_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_dense_4_bias_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_5_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_dense_5_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_28AssignVariableOp4assignvariableop_28_adam_lstm_2_lstm_cell_3_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_29AssignVariableOp>assignvariableop_29_adam_lstm_2_lstm_cell_3_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_30AssignVariableOp2assignvariableop_30_adam_lstm_2_lstm_cell_3_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ù
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: æ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_30AssignVariableOp_302(
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
æÇ
Þ
A__inference_lstm_2_layer_call_and_return_conditional_losses_24544
inputs_0<
)lstm_cell_3_split_readvariableop_resource:	d:
+lstm_cell_3_split_1_readvariableop_resource:	6
#lstm_cell_3_readvariableop_resource:	@
identity¢lstm_cell_3/ReadVariableOp¢lstm_cell_3/ReadVariableOp_1¢lstm_cell_3/ReadVariableOp_2¢lstm_cell_3/ReadVariableOp_3¢ lstm_cell_3/split/ReadVariableOp¢"lstm_cell_3/split_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskc
lstm_cell_3/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:`
lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_3/ones_likeFill$lstm_cell_3/ones_like/Shape:output:0$lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd^
lstm_cell_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_3/dropout/MulMullstm_cell_3/ones_like:output:0"lstm_cell_3/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
lstm_cell_3/dropout/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:¤
0lstm_cell_3/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_3/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0g
"lstm_cell_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ê
 lstm_cell_3/dropout/GreaterEqualGreaterEqual9lstm_cell_3/dropout/random_uniform/RandomUniform:output:0+lstm_cell_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/dropout/CastCast$lstm_cell_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/dropout/Mul_1Mullstm_cell_3/dropout/Mul:z:0lstm_cell_3/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
lstm_cell_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_3/dropout_1/MulMullstm_cell_3/ones_like:output:0$lstm_cell_3/dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
lstm_cell_3/dropout_1/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:¨
2lstm_cell_3/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0i
$lstm_cell_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ð
"lstm_cell_3/dropout_1/GreaterEqualGreaterEqual;lstm_cell_3/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/dropout_1/CastCast&lstm_cell_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/dropout_1/Mul_1Mullstm_cell_3/dropout_1/Mul:z:0lstm_cell_3/dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
lstm_cell_3/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_3/dropout_2/MulMullstm_cell_3/ones_like:output:0$lstm_cell_3/dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
lstm_cell_3/dropout_2/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:¨
2lstm_cell_3/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0i
$lstm_cell_3/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ð
"lstm_cell_3/dropout_2/GreaterEqualGreaterEqual;lstm_cell_3/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/dropout_2/CastCast&lstm_cell_3/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/dropout_2/Mul_1Mullstm_cell_3/dropout_2/Mul:z:0lstm_cell_3/dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
lstm_cell_3/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_3/dropout_3/MulMullstm_cell_3/ones_like:output:0$lstm_cell_3/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
lstm_cell_3/dropout_3/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:¨
2lstm_cell_3/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0i
$lstm_cell_3/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ð
"lstm_cell_3/dropout_3/GreaterEqualGreaterEqual;lstm_cell_3/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/dropout_3/CastCast&lstm_cell_3/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/dropout_3/Mul_1Mullstm_cell_3/dropout_3/Mul:z:0lstm_cell_3/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
lstm_cell_3/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:b
lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
lstm_cell_3/ones_like_1Fill&lstm_cell_3/ones_like_1/Shape:output:0&lstm_cell_3/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
lstm_cell_3/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_3/dropout_4/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
lstm_cell_3/dropout_4/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:¨
2lstm_cell_3/dropout_4/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0i
$lstm_cell_3/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ð
"lstm_cell_3/dropout_4/GreaterEqualGreaterEqual;lstm_cell_3/dropout_4/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/dropout_4/CastCast&lstm_cell_3/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/dropout_4/Mul_1Mullstm_cell_3/dropout_4/Mul:z:0lstm_cell_3/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
lstm_cell_3/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_3/dropout_5/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
lstm_cell_3/dropout_5/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:¨
2lstm_cell_3/dropout_5/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0i
$lstm_cell_3/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ð
"lstm_cell_3/dropout_5/GreaterEqualGreaterEqual;lstm_cell_3/dropout_5/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/dropout_5/CastCast&lstm_cell_3/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/dropout_5/Mul_1Mullstm_cell_3/dropout_5/Mul:z:0lstm_cell_3/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
lstm_cell_3/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_3/dropout_6/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_6/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
lstm_cell_3/dropout_6/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:¨
2lstm_cell_3/dropout_6/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_6/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0i
$lstm_cell_3/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ð
"lstm_cell_3/dropout_6/GreaterEqualGreaterEqual;lstm_cell_3/dropout_6/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/dropout_6/CastCast&lstm_cell_3/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/dropout_6/Mul_1Mullstm_cell_3/dropout_6/Mul:z:0lstm_cell_3/dropout_6/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
lstm_cell_3/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_3/dropout_7/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_7/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
lstm_cell_3/dropout_7/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:¨
2lstm_cell_3/dropout_7/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_7/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0i
$lstm_cell_3/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ð
"lstm_cell_3/dropout_7/GreaterEqualGreaterEqual;lstm_cell_3/dropout_7/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/dropout_7/CastCast&lstm_cell_3/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/dropout_7/Mul_1Mullstm_cell_3/dropout_7/Mul:z:0lstm_cell_3/dropout_7/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/mulMulstrided_slice_2:output:0lstm_cell_3/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/mul_1Mulstrided_slice_2:output:0lstm_cell_3/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/mul_2Mulstrided_slice_2:output:0lstm_cell_3/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/mul_3Mulstrided_slice_2:output:0lstm_cell_3/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_cell_3/split/ReadVariableOpReadVariableOp)lstm_cell_3_split_readvariableop_resource*
_output_shapes
:	d*
dtype0Â
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0(lstm_cell_3/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:d@:d@:d@:d@*
	num_split
lstm_cell_3/MatMulMatMullstm_cell_3/mul:z:0lstm_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/MatMul_1MatMullstm_cell_3/mul_1:z:0lstm_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/MatMul_2MatMullstm_cell_3/mul_2:z:0lstm_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/MatMul_3MatMullstm_cell_3/mul_3:z:0lstm_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
"lstm_cell_3/split_1/ReadVariableOpReadVariableOp+lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0¸
lstm_cell_3/split_1Split&lstm_cell_3/split_1/split_dim:output:0*lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split
lstm_cell_3/BiasAddBiasAddlstm_cell_3/MatMul:product:0lstm_cell_3/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/BiasAdd_1BiasAddlstm_cell_3/MatMul_1:product:0lstm_cell_3/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/BiasAdd_2BiasAddlstm_cell_3/MatMul_2:product:0lstm_cell_3/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/BiasAdd_3BiasAddlstm_cell_3/MatMul_3:product:0lstm_cell_3/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
lstm_cell_3/mul_4Mulzeros:output:0lstm_cell_3/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
lstm_cell_3/mul_5Mulzeros:output:0lstm_cell_3/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
lstm_cell_3/mul_6Mulzeros:output:0lstm_cell_3/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
lstm_cell_3/mul_7Mulzeros:output:0lstm_cell_3/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/ReadVariableOpReadVariableOp#lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0p
lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   r
!lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
lstm_cell_3/strided_sliceStridedSlice"lstm_cell_3/ReadVariableOp:value:0(lstm_cell_3/strided_slice/stack:output:0*lstm_cell_3/strided_slice/stack_1:output:0*lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
lstm_cell_3/MatMul_4MatMullstm_cell_3/mul_4:z:0"lstm_cell_3/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/addAddV2lstm_cell_3/BiasAdd:output:0lstm_cell_3/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
lstm_cell_3/SigmoidSigmoidlstm_cell_3/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/ReadVariableOp_1ReadVariableOp#lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0r
!lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   t
#lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ±
lstm_cell_3/strided_slice_1StridedSlice$lstm_cell_3/ReadVariableOp_1:value:0*lstm_cell_3/strided_slice_1/stack:output:0,lstm_cell_3/strided_slice_1/stack_1:output:0,lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
lstm_cell_3/MatMul_5MatMullstm_cell_3/mul_5:z:0$lstm_cell_3/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/add_1AddV2lstm_cell_3/BiasAdd_1:output:0lstm_cell_3/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
lstm_cell_3/mul_8Mullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/ReadVariableOp_2ReadVariableOp#lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0r
!lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   t
#lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ±
lstm_cell_3/strided_slice_2StridedSlice$lstm_cell_3/ReadVariableOp_2:value:0*lstm_cell_3/strided_slice_2/stack:output:0,lstm_cell_3/strided_slice_2/stack_1:output:0,lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
lstm_cell_3/MatMul_6MatMullstm_cell_3/mul_6:z:0$lstm_cell_3/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/add_2AddV2lstm_cell_3/BiasAdd_2:output:0lstm_cell_3/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
lstm_cell_3/TanhTanhlstm_cell_3/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
lstm_cell_3/mul_9Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
lstm_cell_3/add_3AddV2lstm_cell_3/mul_8:z:0lstm_cell_3/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/ReadVariableOp_3ReadVariableOp#lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0r
!lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   t
#lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ±
lstm_cell_3/strided_slice_3StridedSlice$lstm_cell_3/ReadVariableOp_3:value:0*lstm_cell_3/strided_slice_3/stack:output:0,lstm_cell_3/strided_slice_3/stack_1:output:0,lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
lstm_cell_3/MatMul_7MatMullstm_cell_3/mul_7:z:0$lstm_cell_3/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/add_4AddV2lstm_cell_3/BiasAdd_3:output:0lstm_cell_3/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
lstm_cell_3/Tanh_1Tanhlstm_cell_3/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
lstm_cell_3/mul_10Mullstm_cell_3/Sigmoid_2:y:0lstm_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ó
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_3_split_readvariableop_resource+lstm_cell_3_split_1_readvariableop_resource#lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_24346*
condR
while_cond_24345*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^lstm_cell_3/ReadVariableOp^lstm_cell_3/ReadVariableOp_1^lstm_cell_3/ReadVariableOp_2^lstm_cell_3/ReadVariableOp_3!^lstm_cell_3/split/ReadVariableOp#^lstm_cell_3/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: : : 28
lstm_cell_3/ReadVariableOplstm_cell_3/ReadVariableOp2<
lstm_cell_3/ReadVariableOp_1lstm_cell_3/ReadVariableOp_12<
lstm_cell_3/ReadVariableOp_2lstm_cell_3/ReadVariableOp_22<
lstm_cell_3/ReadVariableOp_3lstm_cell_3/ReadVariableOp_32D
 lstm_cell_3/split/ReadVariableOp lstm_cell_3/split/ReadVariableOp2H
"lstm_cell_3/split_1/ReadVariableOp"lstm_cell_3/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
inputs/0
Â
	
while_body_24960
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_3_split_readvariableop_resource_0:	dB
3while_lstm_cell_3_split_1_readvariableop_resource_0:	>
+while_lstm_cell_3_readvariableop_resource_0:	@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_3_split_readvariableop_resource:	d@
1while_lstm_cell_3_split_1_readvariableop_resource:	<
)while_lstm_cell_3_readvariableop_resource:	@¢ while/lstm_cell_3/ReadVariableOp¢"while/lstm_cell_3/ReadVariableOp_1¢"while/lstm_cell_3/ReadVariableOp_2¢"while/lstm_cell_3/ReadVariableOp_3¢&while/lstm_cell_3/split/ReadVariableOp¢(while/lstm_cell_3/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
!while/lstm_cell_3/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?­
while/lstm_cell_3/ones_likeFill*while/lstm_cell_3/ones_like/Shape:output:0*while/lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
while/lstm_cell_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¦
while/lstm_cell_3/dropout/MulMul$while/lstm_cell_3/ones_like:output:0(while/lstm_cell_3/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/lstm_cell_3/dropout/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:°
6while/lstm_cell_3/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_3/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0m
(while/lstm_cell_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ü
&while/lstm_cell_3/dropout/GreaterEqualGreaterEqual?while/lstm_cell_3/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_3/dropout/CastCast*while/lstm_cell_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_3/dropout/Mul_1Mul!while/lstm_cell_3/dropout/Mul:z:0"while/lstm_cell_3/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
!while/lstm_cell_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?ª
while/lstm_cell_3/dropout_1/MulMul$while/lstm_cell_3/ones_like:output:0*while/lstm_cell_3/dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
!while/lstm_cell_3/dropout_1/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:´
8while/lstm_cell_3/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0o
*while/lstm_cell_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=â
(while/lstm_cell_3/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 while/lstm_cell_3/dropout_1/CastCast,while/lstm_cell_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¥
!while/lstm_cell_3/dropout_1/Mul_1Mul#while/lstm_cell_3/dropout_1/Mul:z:0$while/lstm_cell_3/dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
!while/lstm_cell_3/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?ª
while/lstm_cell_3/dropout_2/MulMul$while/lstm_cell_3/ones_like:output:0*while/lstm_cell_3/dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
!while/lstm_cell_3/dropout_2/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:´
8while/lstm_cell_3/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0o
*while/lstm_cell_3/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=â
(while/lstm_cell_3/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 while/lstm_cell_3/dropout_2/CastCast,while/lstm_cell_3/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¥
!while/lstm_cell_3/dropout_2/Mul_1Mul#while/lstm_cell_3/dropout_2/Mul:z:0$while/lstm_cell_3/dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
!while/lstm_cell_3/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?ª
while/lstm_cell_3/dropout_3/MulMul$while/lstm_cell_3/ones_like:output:0*while/lstm_cell_3/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
!while/lstm_cell_3/dropout_3/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:´
8while/lstm_cell_3/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0o
*while/lstm_cell_3/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=â
(while/lstm_cell_3/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 while/lstm_cell_3/dropout_3/CastCast,while/lstm_cell_3/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¥
!while/lstm_cell_3/dropout_3/Mul_1Mul#while/lstm_cell_3/dropout_3/Mul:z:0$while/lstm_cell_3/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
#while/lstm_cell_3/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:h
#while/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
while/lstm_cell_3/ones_like_1Fill,while/lstm_cell_3/ones_like_1/Shape:output:0,while/lstm_cell_3/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@f
!while/lstm_cell_3/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¬
while/lstm_cell_3/dropout_4/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
!while/lstm_cell_3/dropout_4/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:´
8while/lstm_cell_3/dropout_4/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0o
*while/lstm_cell_3/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=â
(while/lstm_cell_3/dropout_4/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_4/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 while/lstm_cell_3/dropout_4/CastCast,while/lstm_cell_3/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
!while/lstm_cell_3/dropout_4/Mul_1Mul#while/lstm_cell_3/dropout_4/Mul:z:0$while/lstm_cell_3/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@f
!while/lstm_cell_3/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¬
while/lstm_cell_3/dropout_5/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
!while/lstm_cell_3/dropout_5/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:´
8while/lstm_cell_3/dropout_5/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0o
*while/lstm_cell_3/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=â
(while/lstm_cell_3/dropout_5/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_5/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 while/lstm_cell_3/dropout_5/CastCast,while/lstm_cell_3/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
!while/lstm_cell_3/dropout_5/Mul_1Mul#while/lstm_cell_3/dropout_5/Mul:z:0$while/lstm_cell_3/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@f
!while/lstm_cell_3/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¬
while/lstm_cell_3/dropout_6/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_6/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
!while/lstm_cell_3/dropout_6/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:´
8while/lstm_cell_3/dropout_6/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_6/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0o
*while/lstm_cell_3/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=â
(while/lstm_cell_3/dropout_6/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_6/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 while/lstm_cell_3/dropout_6/CastCast,while/lstm_cell_3/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
!while/lstm_cell_3/dropout_6/Mul_1Mul#while/lstm_cell_3/dropout_6/Mul:z:0$while/lstm_cell_3/dropout_6/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@f
!while/lstm_cell_3/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¬
while/lstm_cell_3/dropout_7/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_7/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
!while/lstm_cell_3/dropout_7/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:´
8while/lstm_cell_3/dropout_7/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_7/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0o
*while/lstm_cell_3/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=â
(while/lstm_cell_3/dropout_7/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_7/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 while/lstm_cell_3/dropout_7/CastCast,while/lstm_cell_3/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
!while/lstm_cell_3/dropout_7/Mul_1Mul#while/lstm_cell_3/dropout_7/Mul:z:0$while/lstm_cell_3/dropout_7/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
while/lstm_cell_3/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell_3/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd©
while/lstm_cell_3/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_3/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd©
while/lstm_cell_3/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_3/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd©
while/lstm_cell_3/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_3/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
&while/lstm_cell_3/split/ReadVariableOpReadVariableOp1while_lstm_cell_3_split_readvariableop_resource_0*
_output_shapes
:	d*
dtype0Ô
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0.while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:d@:d@:d@:d@*
	num_split
while/lstm_cell_3/MatMulMatMulwhile/lstm_cell_3/mul:z:0 while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/MatMul_1MatMulwhile/lstm_cell_3/mul_1:z:0 while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/MatMul_2MatMulwhile/lstm_cell_3/mul_2:z:0 while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/MatMul_3MatMulwhile/lstm_cell_3/mul_3:z:0 while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
#while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
(while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Ê
while/lstm_cell_3/split_1Split,while/lstm_cell_3/split_1/split_dim:output:00while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split
while/lstm_cell_3/BiasAddBiasAdd"while/lstm_cell_3/MatMul:product:0"while/lstm_cell_3/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
while/lstm_cell_3/BiasAdd_1BiasAdd$while/lstm_cell_3/MatMul_1:product:0"while/lstm_cell_3/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
while/lstm_cell_3/BiasAdd_2BiasAdd$while/lstm_cell_3/MatMul_2:product:0"while/lstm_cell_3/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
while/lstm_cell_3/BiasAdd_3BiasAdd$while/lstm_cell_3/MatMul_3:product:0"while/lstm_cell_3/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_4Mulwhile_placeholder_2%while/lstm_cell_3/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_5Mulwhile_placeholder_2%while/lstm_cell_3/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_6Mulwhile_placeholder_2%while/lstm_cell_3/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_7Mulwhile_placeholder_2%while/lstm_cell_3/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 while/lstm_cell_3/ReadVariableOpReadVariableOp+while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0v
%while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   x
'while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Å
while/lstm_cell_3/strided_sliceStridedSlice(while/lstm_cell_3/ReadVariableOp:value:0.while/lstm_cell_3/strided_slice/stack:output:00while/lstm_cell_3/strided_slice/stack_1:output:00while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_4MatMulwhile/lstm_cell_3/mul_4:z:0(while/lstm_cell_3/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/addAddV2"while/lstm_cell_3/BiasAdd:output:0$while/lstm_cell_3/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
while/lstm_cell_3/SigmoidSigmoidwhile/lstm_cell_3/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"while/lstm_cell_3/ReadVariableOp_1ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0x
'while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   z
)while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ï
!while/lstm_cell_3/strided_slice_1StridedSlice*while/lstm_cell_3/ReadVariableOp_1:value:00while/lstm_cell_3/strided_slice_1/stack:output:02while/lstm_cell_3/strided_slice_1/stack_1:output:02while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_5MatMulwhile/lstm_cell_3/mul_5:z:0*while/lstm_cell_3/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/add_1AddV2$while/lstm_cell_3/BiasAdd_1:output:0$while/lstm_cell_3/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@u
while/lstm_cell_3/Sigmoid_1Sigmoidwhile/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_8Mulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"while/lstm_cell_3/ReadVariableOp_2ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0x
'while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   z
)while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ï
!while/lstm_cell_3/strided_slice_2StridedSlice*while/lstm_cell_3/ReadVariableOp_2:value:00while/lstm_cell_3/strided_slice_2/stack:output:02while/lstm_cell_3/strided_slice_2/stack_1:output:02while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_6MatMulwhile/lstm_cell_3/mul_6:z:0*while/lstm_cell_3/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/add_2AddV2$while/lstm_cell_3/BiasAdd_2:output:0$while/lstm_cell_3/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
while/lstm_cell_3/TanhTanhwhile/lstm_cell_3/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_9Mulwhile/lstm_cell_3/Sigmoid:y:0while/lstm_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/add_3AddV2while/lstm_cell_3/mul_8:z:0while/lstm_cell_3/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"while/lstm_cell_3/ReadVariableOp_3ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0x
'while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   z
)while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ï
!while/lstm_cell_3/strided_slice_3StridedSlice*while/lstm_cell_3/ReadVariableOp_3:value:00while/lstm_cell_3/strided_slice_3/stack:output:02while/lstm_cell_3/strided_slice_3/stack_1:output:02while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_7MatMulwhile/lstm_cell_3/mul_7:z:0*while/lstm_cell_3/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/add_4AddV2$while/lstm_cell_3/BiasAdd_3:output:0$while/lstm_cell_3/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@u
while/lstm_cell_3/Sigmoid_2Sigmoidwhile/lstm_cell_3/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
while/lstm_cell_3/Tanh_1Tanhwhile/lstm_cell_3/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_10Mulwhile/lstm_cell_3/Sigmoid_2:y:0while/lstm_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_10:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒy
while/Identity_4Identitywhile/lstm_cell_3/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
while/Identity_5Identitywhile/lstm_cell_3/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@²

while/NoOpNoOp!^while/lstm_cell_3/ReadVariableOp#^while/lstm_cell_3/ReadVariableOp_1#^while/lstm_cell_3/ReadVariableOp_2#^while/lstm_cell_3/ReadVariableOp_3'^while/lstm_cell_3/split/ReadVariableOp)^while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_3_readvariableop_resource+while_lstm_cell_3_readvariableop_resource_0"h
1while_lstm_cell_3_split_1_readvariableop_resource3while_lstm_cell_3_split_1_readvariableop_resource_0"d
/while_lstm_cell_3_split_readvariableop_resource1while_lstm_cell_3_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : 2D
 while/lstm_cell_3/ReadVariableOp while/lstm_cell_3/ReadVariableOp2H
"while/lstm_cell_3/ReadVariableOp_1"while/lstm_cell_3/ReadVariableOp_12H
"while/lstm_cell_3/ReadVariableOp_2"while/lstm_cell_3/ReadVariableOp_22H
"while/lstm_cell_3/ReadVariableOp_3"while/lstm_cell_3/ReadVariableOp_32P
&while/lstm_cell_3/split/ReadVariableOp&while/lstm_cell_3/split/ReadVariableOp2T
(while/lstm_cell_3/split_1/ReadVariableOp(while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 

Ü
A__inference_lstm_2_layer_call_and_return_conditional_losses_22526

inputs<
)lstm_cell_3_split_readvariableop_resource:	d:
+lstm_cell_3_split_1_readvariableop_resource:	6
#lstm_cell_3_readvariableop_resource:	@
identity¢lstm_cell_3/ReadVariableOp¢lstm_cell_3/ReadVariableOp_1¢lstm_cell_3/ReadVariableOp_2¢lstm_cell_3/ReadVariableOp_3¢ lstm_cell_3/split/ReadVariableOp¢"lstm_cell_3/split_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:úÿÿÿÿÿÿÿÿÿdD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskc
lstm_cell_3/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:`
lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_3/ones_likeFill$lstm_cell_3/ones_like/Shape:output:0$lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
lstm_cell_3/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:b
lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
lstm_cell_3/ones_like_1Fill&lstm_cell_3/ones_like_1/Shape:output:0&lstm_cell_3/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/mulMulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/mul_1Mulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/mul_2Mulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/mul_3Mulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_cell_3/split/ReadVariableOpReadVariableOp)lstm_cell_3_split_readvariableop_resource*
_output_shapes
:	d*
dtype0Â
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0(lstm_cell_3/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:d@:d@:d@:d@*
	num_split
lstm_cell_3/MatMulMatMullstm_cell_3/mul:z:0lstm_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/MatMul_1MatMullstm_cell_3/mul_1:z:0lstm_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/MatMul_2MatMullstm_cell_3/mul_2:z:0lstm_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/MatMul_3MatMullstm_cell_3/mul_3:z:0lstm_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
"lstm_cell_3/split_1/ReadVariableOpReadVariableOp+lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0¸
lstm_cell_3/split_1Split&lstm_cell_3/split_1/split_dim:output:0*lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split
lstm_cell_3/BiasAddBiasAddlstm_cell_3/MatMul:product:0lstm_cell_3/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/BiasAdd_1BiasAddlstm_cell_3/MatMul_1:product:0lstm_cell_3/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/BiasAdd_2BiasAddlstm_cell_3/MatMul_2:product:0lstm_cell_3/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/BiasAdd_3BiasAddlstm_cell_3/MatMul_3:product:0lstm_cell_3/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
lstm_cell_3/mul_4Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
lstm_cell_3/mul_5Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
lstm_cell_3/mul_6Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
lstm_cell_3/mul_7Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/ReadVariableOpReadVariableOp#lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0p
lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   r
!lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
lstm_cell_3/strided_sliceStridedSlice"lstm_cell_3/ReadVariableOp:value:0(lstm_cell_3/strided_slice/stack:output:0*lstm_cell_3/strided_slice/stack_1:output:0*lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
lstm_cell_3/MatMul_4MatMullstm_cell_3/mul_4:z:0"lstm_cell_3/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/addAddV2lstm_cell_3/BiasAdd:output:0lstm_cell_3/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
lstm_cell_3/SigmoidSigmoidlstm_cell_3/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/ReadVariableOp_1ReadVariableOp#lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0r
!lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   t
#lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ±
lstm_cell_3/strided_slice_1StridedSlice$lstm_cell_3/ReadVariableOp_1:value:0*lstm_cell_3/strided_slice_1/stack:output:0,lstm_cell_3/strided_slice_1/stack_1:output:0,lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
lstm_cell_3/MatMul_5MatMullstm_cell_3/mul_5:z:0$lstm_cell_3/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/add_1AddV2lstm_cell_3/BiasAdd_1:output:0lstm_cell_3/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
lstm_cell_3/mul_8Mullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/ReadVariableOp_2ReadVariableOp#lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0r
!lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   t
#lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ±
lstm_cell_3/strided_slice_2StridedSlice$lstm_cell_3/ReadVariableOp_2:value:0*lstm_cell_3/strided_slice_2/stack:output:0,lstm_cell_3/strided_slice_2/stack_1:output:0,lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
lstm_cell_3/MatMul_6MatMullstm_cell_3/mul_6:z:0$lstm_cell_3/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/add_2AddV2lstm_cell_3/BiasAdd_2:output:0lstm_cell_3/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
lstm_cell_3/TanhTanhlstm_cell_3/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
lstm_cell_3/mul_9Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
lstm_cell_3/add_3AddV2lstm_cell_3/mul_8:z:0lstm_cell_3/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/ReadVariableOp_3ReadVariableOp#lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0r
!lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   t
#lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ±
lstm_cell_3/strided_slice_3StridedSlice$lstm_cell_3/ReadVariableOp_3:value:0*lstm_cell_3/strided_slice_3/stack:output:0,lstm_cell_3/strided_slice_3/stack_1:output:0,lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
lstm_cell_3/MatMul_7MatMullstm_cell_3/mul_7:z:0$lstm_cell_3/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/add_4AddV2lstm_cell_3/BiasAdd_3:output:0lstm_cell_3/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
lstm_cell_3/Tanh_1Tanhlstm_cell_3/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
lstm_cell_3/mul_10Mullstm_cell_3/Sigmoid_2:y:0lstm_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ó
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_3_split_readvariableop_resource+lstm_cell_3_split_1_readvariableop_resource#lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_22392*
condR
while_cond_22391*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ã
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:úÿÿÿÿÿÿÿÿÿ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿú@[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^lstm_cell_3/ReadVariableOp^lstm_cell_3/ReadVariableOp_1^lstm_cell_3/ReadVariableOp_2^lstm_cell_3/ReadVariableOp_3!^lstm_cell_3/split/ReadVariableOp#^lstm_cell_3/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿúd: : : 28
lstm_cell_3/ReadVariableOplstm_cell_3/ReadVariableOp2<
lstm_cell_3/ReadVariableOp_1lstm_cell_3/ReadVariableOp_12<
lstm_cell_3/ReadVariableOp_2lstm_cell_3/ReadVariableOp_22<
lstm_cell_3/ReadVariableOp_3lstm_cell_3/ReadVariableOp_32D
 lstm_cell_3/split/ReadVariableOp lstm_cell_3/split/ReadVariableOp2H
"lstm_cell_3/split_1/ReadVariableOp"lstm_cell_3/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúd
 
_user_specified_nameinputs
«

+__inference_embedding_4_layer_call_fn_23876

inputs
unknown:	Äd
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúd*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_embedding_4_layer_call_and_return_conditional_losses_22280t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:ÿÿÿÿÿÿÿÿÿú: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs
Â
	
while_body_24346
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_3_split_readvariableop_resource_0:	dB
3while_lstm_cell_3_split_1_readvariableop_resource_0:	>
+while_lstm_cell_3_readvariableop_resource_0:	@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_3_split_readvariableop_resource:	d@
1while_lstm_cell_3_split_1_readvariableop_resource:	<
)while_lstm_cell_3_readvariableop_resource:	@¢ while/lstm_cell_3/ReadVariableOp¢"while/lstm_cell_3/ReadVariableOp_1¢"while/lstm_cell_3/ReadVariableOp_2¢"while/lstm_cell_3/ReadVariableOp_3¢&while/lstm_cell_3/split/ReadVariableOp¢(while/lstm_cell_3/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
!while/lstm_cell_3/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?­
while/lstm_cell_3/ones_likeFill*while/lstm_cell_3/ones_like/Shape:output:0*while/lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
while/lstm_cell_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¦
while/lstm_cell_3/dropout/MulMul$while/lstm_cell_3/ones_like:output:0(while/lstm_cell_3/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/lstm_cell_3/dropout/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:°
6while/lstm_cell_3/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_3/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0m
(while/lstm_cell_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ü
&while/lstm_cell_3/dropout/GreaterEqualGreaterEqual?while/lstm_cell_3/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_3/dropout/CastCast*while/lstm_cell_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_3/dropout/Mul_1Mul!while/lstm_cell_3/dropout/Mul:z:0"while/lstm_cell_3/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
!while/lstm_cell_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?ª
while/lstm_cell_3/dropout_1/MulMul$while/lstm_cell_3/ones_like:output:0*while/lstm_cell_3/dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
!while/lstm_cell_3/dropout_1/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:´
8while/lstm_cell_3/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0o
*while/lstm_cell_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=â
(while/lstm_cell_3/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 while/lstm_cell_3/dropout_1/CastCast,while/lstm_cell_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¥
!while/lstm_cell_3/dropout_1/Mul_1Mul#while/lstm_cell_3/dropout_1/Mul:z:0$while/lstm_cell_3/dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
!while/lstm_cell_3/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?ª
while/lstm_cell_3/dropout_2/MulMul$while/lstm_cell_3/ones_like:output:0*while/lstm_cell_3/dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
!while/lstm_cell_3/dropout_2/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:´
8while/lstm_cell_3/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0o
*while/lstm_cell_3/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=â
(while/lstm_cell_3/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 while/lstm_cell_3/dropout_2/CastCast,while/lstm_cell_3/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¥
!while/lstm_cell_3/dropout_2/Mul_1Mul#while/lstm_cell_3/dropout_2/Mul:z:0$while/lstm_cell_3/dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
!while/lstm_cell_3/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?ª
while/lstm_cell_3/dropout_3/MulMul$while/lstm_cell_3/ones_like:output:0*while/lstm_cell_3/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
!while/lstm_cell_3/dropout_3/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:´
8while/lstm_cell_3/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0o
*while/lstm_cell_3/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=â
(while/lstm_cell_3/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 while/lstm_cell_3/dropout_3/CastCast,while/lstm_cell_3/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¥
!while/lstm_cell_3/dropout_3/Mul_1Mul#while/lstm_cell_3/dropout_3/Mul:z:0$while/lstm_cell_3/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
#while/lstm_cell_3/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:h
#while/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
while/lstm_cell_3/ones_like_1Fill,while/lstm_cell_3/ones_like_1/Shape:output:0,while/lstm_cell_3/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@f
!while/lstm_cell_3/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¬
while/lstm_cell_3/dropout_4/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
!while/lstm_cell_3/dropout_4/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:´
8while/lstm_cell_3/dropout_4/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0o
*while/lstm_cell_3/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=â
(while/lstm_cell_3/dropout_4/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_4/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 while/lstm_cell_3/dropout_4/CastCast,while/lstm_cell_3/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
!while/lstm_cell_3/dropout_4/Mul_1Mul#while/lstm_cell_3/dropout_4/Mul:z:0$while/lstm_cell_3/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@f
!while/lstm_cell_3/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¬
while/lstm_cell_3/dropout_5/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
!while/lstm_cell_3/dropout_5/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:´
8while/lstm_cell_3/dropout_5/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0o
*while/lstm_cell_3/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=â
(while/lstm_cell_3/dropout_5/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_5/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 while/lstm_cell_3/dropout_5/CastCast,while/lstm_cell_3/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
!while/lstm_cell_3/dropout_5/Mul_1Mul#while/lstm_cell_3/dropout_5/Mul:z:0$while/lstm_cell_3/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@f
!while/lstm_cell_3/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¬
while/lstm_cell_3/dropout_6/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_6/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
!while/lstm_cell_3/dropout_6/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:´
8while/lstm_cell_3/dropout_6/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_6/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0o
*while/lstm_cell_3/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=â
(while/lstm_cell_3/dropout_6/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_6/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 while/lstm_cell_3/dropout_6/CastCast,while/lstm_cell_3/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
!while/lstm_cell_3/dropout_6/Mul_1Mul#while/lstm_cell_3/dropout_6/Mul:z:0$while/lstm_cell_3/dropout_6/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@f
!while/lstm_cell_3/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¬
while/lstm_cell_3/dropout_7/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_7/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
!while/lstm_cell_3/dropout_7/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:´
8while/lstm_cell_3/dropout_7/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_7/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0o
*while/lstm_cell_3/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=â
(while/lstm_cell_3/dropout_7/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_7/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 while/lstm_cell_3/dropout_7/CastCast,while/lstm_cell_3/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
!while/lstm_cell_3/dropout_7/Mul_1Mul#while/lstm_cell_3/dropout_7/Mul:z:0$while/lstm_cell_3/dropout_7/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
while/lstm_cell_3/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell_3/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd©
while/lstm_cell_3/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_3/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd©
while/lstm_cell_3/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_3/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd©
while/lstm_cell_3/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_3/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
&while/lstm_cell_3/split/ReadVariableOpReadVariableOp1while_lstm_cell_3_split_readvariableop_resource_0*
_output_shapes
:	d*
dtype0Ô
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0.while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:d@:d@:d@:d@*
	num_split
while/lstm_cell_3/MatMulMatMulwhile/lstm_cell_3/mul:z:0 while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/MatMul_1MatMulwhile/lstm_cell_3/mul_1:z:0 while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/MatMul_2MatMulwhile/lstm_cell_3/mul_2:z:0 while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/MatMul_3MatMulwhile/lstm_cell_3/mul_3:z:0 while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
#while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
(while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Ê
while/lstm_cell_3/split_1Split,while/lstm_cell_3/split_1/split_dim:output:00while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split
while/lstm_cell_3/BiasAddBiasAdd"while/lstm_cell_3/MatMul:product:0"while/lstm_cell_3/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
while/lstm_cell_3/BiasAdd_1BiasAdd$while/lstm_cell_3/MatMul_1:product:0"while/lstm_cell_3/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
while/lstm_cell_3/BiasAdd_2BiasAdd$while/lstm_cell_3/MatMul_2:product:0"while/lstm_cell_3/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
while/lstm_cell_3/BiasAdd_3BiasAdd$while/lstm_cell_3/MatMul_3:product:0"while/lstm_cell_3/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_4Mulwhile_placeholder_2%while/lstm_cell_3/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_5Mulwhile_placeholder_2%while/lstm_cell_3/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_6Mulwhile_placeholder_2%while/lstm_cell_3/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_7Mulwhile_placeholder_2%while/lstm_cell_3/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 while/lstm_cell_3/ReadVariableOpReadVariableOp+while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0v
%while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   x
'while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Å
while/lstm_cell_3/strided_sliceStridedSlice(while/lstm_cell_3/ReadVariableOp:value:0.while/lstm_cell_3/strided_slice/stack:output:00while/lstm_cell_3/strided_slice/stack_1:output:00while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_4MatMulwhile/lstm_cell_3/mul_4:z:0(while/lstm_cell_3/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/addAddV2"while/lstm_cell_3/BiasAdd:output:0$while/lstm_cell_3/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
while/lstm_cell_3/SigmoidSigmoidwhile/lstm_cell_3/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"while/lstm_cell_3/ReadVariableOp_1ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0x
'while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   z
)while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ï
!while/lstm_cell_3/strided_slice_1StridedSlice*while/lstm_cell_3/ReadVariableOp_1:value:00while/lstm_cell_3/strided_slice_1/stack:output:02while/lstm_cell_3/strided_slice_1/stack_1:output:02while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_5MatMulwhile/lstm_cell_3/mul_5:z:0*while/lstm_cell_3/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/add_1AddV2$while/lstm_cell_3/BiasAdd_1:output:0$while/lstm_cell_3/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@u
while/lstm_cell_3/Sigmoid_1Sigmoidwhile/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_8Mulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"while/lstm_cell_3/ReadVariableOp_2ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0x
'while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   z
)while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ï
!while/lstm_cell_3/strided_slice_2StridedSlice*while/lstm_cell_3/ReadVariableOp_2:value:00while/lstm_cell_3/strided_slice_2/stack:output:02while/lstm_cell_3/strided_slice_2/stack_1:output:02while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_6MatMulwhile/lstm_cell_3/mul_6:z:0*while/lstm_cell_3/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/add_2AddV2$while/lstm_cell_3/BiasAdd_2:output:0$while/lstm_cell_3/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
while/lstm_cell_3/TanhTanhwhile/lstm_cell_3/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_9Mulwhile/lstm_cell_3/Sigmoid:y:0while/lstm_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/add_3AddV2while/lstm_cell_3/mul_8:z:0while/lstm_cell_3/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"while/lstm_cell_3/ReadVariableOp_3ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0x
'while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   z
)while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ï
!while/lstm_cell_3/strided_slice_3StridedSlice*while/lstm_cell_3/ReadVariableOp_3:value:00while/lstm_cell_3/strided_slice_3/stack:output:02while/lstm_cell_3/strided_slice_3/stack_1:output:02while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_7MatMulwhile/lstm_cell_3/mul_7:z:0*while/lstm_cell_3/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/add_4AddV2$while/lstm_cell_3/BiasAdd_3:output:0$while/lstm_cell_3/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@u
while/lstm_cell_3/Sigmoid_2Sigmoidwhile/lstm_cell_3/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
while/lstm_cell_3/Tanh_1Tanhwhile/lstm_cell_3/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_10Mulwhile/lstm_cell_3/Sigmoid_2:y:0while/lstm_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_10:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒy
while/Identity_4Identitywhile/lstm_cell_3/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
while/Identity_5Identitywhile/lstm_cell_3/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@²

while/NoOpNoOp!^while/lstm_cell_3/ReadVariableOp#^while/lstm_cell_3/ReadVariableOp_1#^while/lstm_cell_3/ReadVariableOp_2#^while/lstm_cell_3/ReadVariableOp_3'^while/lstm_cell_3/split/ReadVariableOp)^while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_3_readvariableop_resource+while_lstm_cell_3_readvariableop_resource_0"h
1while_lstm_cell_3_split_1_readvariableop_resource3while_lstm_cell_3_split_1_readvariableop_resource_0"d
/while_lstm_cell_3_split_readvariableop_resource1while_lstm_cell_3_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : 2D
 while/lstm_cell_3/ReadVariableOp while/lstm_cell_3/ReadVariableOp2H
"while/lstm_cell_3/ReadVariableOp_1"while/lstm_cell_3/ReadVariableOp_12H
"while/lstm_cell_3/ReadVariableOp_2"while/lstm_cell_3/ReadVariableOp_22H
"while/lstm_cell_3/ReadVariableOp_3"while/lstm_cell_3/ReadVariableOp_32P
&while/lstm_cell_3/split/ReadVariableOp&while/lstm_cell_3/split/ReadVariableOp2T
(while/lstm_cell_3/split_1/ReadVariableOp(while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
¾

'__inference_dense_5_layer_call_fn_25187

inputs
unknown: 
	unknown_0:
identity¢StatefulPartitionedCall×
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
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_22562o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ï	
Î
,__inference_sequential_4_layer_call_fn_23096
embedding_4_input
unknown:	Äd
	unknown_0:	d
	unknown_1:	
	unknown_2:	@
	unknown_3:@ 
	unknown_4: 
	unknown_5: 
	unknown_6:
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallembedding_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_23056o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿú: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
+
_user_specified_nameembedding_4_input
ì
Ó
$sequential_4_lstm_2_while_body_21601D
@sequential_4_lstm_2_while_sequential_4_lstm_2_while_loop_counterJ
Fsequential_4_lstm_2_while_sequential_4_lstm_2_while_maximum_iterations)
%sequential_4_lstm_2_while_placeholder+
'sequential_4_lstm_2_while_placeholder_1+
'sequential_4_lstm_2_while_placeholder_2+
'sequential_4_lstm_2_while_placeholder_3C
?sequential_4_lstm_2_while_sequential_4_lstm_2_strided_slice_1_0
{sequential_4_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_2_tensorarrayunstack_tensorlistfromtensor_0X
Esequential_4_lstm_2_while_lstm_cell_3_split_readvariableop_resource_0:	dV
Gsequential_4_lstm_2_while_lstm_cell_3_split_1_readvariableop_resource_0:	R
?sequential_4_lstm_2_while_lstm_cell_3_readvariableop_resource_0:	@&
"sequential_4_lstm_2_while_identity(
$sequential_4_lstm_2_while_identity_1(
$sequential_4_lstm_2_while_identity_2(
$sequential_4_lstm_2_while_identity_3(
$sequential_4_lstm_2_while_identity_4(
$sequential_4_lstm_2_while_identity_5A
=sequential_4_lstm_2_while_sequential_4_lstm_2_strided_slice_1}
ysequential_4_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_2_tensorarrayunstack_tensorlistfromtensorV
Csequential_4_lstm_2_while_lstm_cell_3_split_readvariableop_resource:	dT
Esequential_4_lstm_2_while_lstm_cell_3_split_1_readvariableop_resource:	P
=sequential_4_lstm_2_while_lstm_cell_3_readvariableop_resource:	@¢4sequential_4/lstm_2/while/lstm_cell_3/ReadVariableOp¢6sequential_4/lstm_2/while/lstm_cell_3/ReadVariableOp_1¢6sequential_4/lstm_2/while/lstm_cell_3/ReadVariableOp_2¢6sequential_4/lstm_2/while/lstm_cell_3/ReadVariableOp_3¢:sequential_4/lstm_2/while/lstm_cell_3/split/ReadVariableOp¢<sequential_4/lstm_2/while/lstm_cell_3/split_1/ReadVariableOp
Ksequential_4/lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   
=sequential_4/lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_4_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_2_tensorarrayunstack_tensorlistfromtensor_0%sequential_4_lstm_2_while_placeholderTsequential_4/lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0©
5sequential_4/lstm_2/while/lstm_cell_3/ones_like/ShapeShapeDsequential_4/lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:z
5sequential_4/lstm_2/while/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?é
/sequential_4/lstm_2/while/lstm_cell_3/ones_likeFill>sequential_4/lstm_2/while/lstm_cell_3/ones_like/Shape:output:0>sequential_4/lstm_2/while/lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
7sequential_4/lstm_2/while/lstm_cell_3/ones_like_1/ShapeShape'sequential_4_lstm_2_while_placeholder_2*
T0*
_output_shapes
:|
7sequential_4/lstm_2/while/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ï
1sequential_4/lstm_2/while/lstm_cell_3/ones_like_1Fill@sequential_4/lstm_2/while/lstm_cell_3/ones_like_1/Shape:output:0@sequential_4/lstm_2/while/lstm_cell_3/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@â
)sequential_4/lstm_2/while/lstm_cell_3/mulMulDsequential_4/lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:08sequential_4/lstm_2/while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdä
+sequential_4/lstm_2/while/lstm_cell_3/mul_1MulDsequential_4/lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:08sequential_4/lstm_2/while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdä
+sequential_4/lstm_2/while/lstm_cell_3/mul_2MulDsequential_4/lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:08sequential_4/lstm_2/while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdä
+sequential_4/lstm_2/while/lstm_cell_3/mul_3MulDsequential_4/lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:08sequential_4/lstm_2/while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
5sequential_4/lstm_2/while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Á
:sequential_4/lstm_2/while/lstm_cell_3/split/ReadVariableOpReadVariableOpEsequential_4_lstm_2_while_lstm_cell_3_split_readvariableop_resource_0*
_output_shapes
:	d*
dtype0
+sequential_4/lstm_2/while/lstm_cell_3/splitSplit>sequential_4/lstm_2/while/lstm_cell_3/split/split_dim:output:0Bsequential_4/lstm_2/while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:d@:d@:d@:d@*
	num_splitÍ
,sequential_4/lstm_2/while/lstm_cell_3/MatMulMatMul-sequential_4/lstm_2/while/lstm_cell_3/mul:z:04sequential_4/lstm_2/while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ñ
.sequential_4/lstm_2/while/lstm_cell_3/MatMul_1MatMul/sequential_4/lstm_2/while/lstm_cell_3/mul_1:z:04sequential_4/lstm_2/while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ñ
.sequential_4/lstm_2/while/lstm_cell_3/MatMul_2MatMul/sequential_4/lstm_2/while/lstm_cell_3/mul_2:z:04sequential_4/lstm_2/while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ñ
.sequential_4/lstm_2/while/lstm_cell_3/MatMul_3MatMul/sequential_4/lstm_2/while/lstm_cell_3/mul_3:z:04sequential_4/lstm_2/while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
7sequential_4/lstm_2/while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Á
<sequential_4/lstm_2/while/lstm_cell_3/split_1/ReadVariableOpReadVariableOpGsequential_4_lstm_2_while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0
-sequential_4/lstm_2/while/lstm_cell_3/split_1Split@sequential_4/lstm_2/while/lstm_cell_3/split_1/split_dim:output:0Dsequential_4/lstm_2/while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitÚ
-sequential_4/lstm_2/while/lstm_cell_3/BiasAddBiasAdd6sequential_4/lstm_2/while/lstm_cell_3/MatMul:product:06sequential_4/lstm_2/while/lstm_cell_3/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Þ
/sequential_4/lstm_2/while/lstm_cell_3/BiasAdd_1BiasAdd8sequential_4/lstm_2/while/lstm_cell_3/MatMul_1:product:06sequential_4/lstm_2/while/lstm_cell_3/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Þ
/sequential_4/lstm_2/while/lstm_cell_3/BiasAdd_2BiasAdd8sequential_4/lstm_2/while/lstm_cell_3/MatMul_2:product:06sequential_4/lstm_2/while/lstm_cell_3/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Þ
/sequential_4/lstm_2/while/lstm_cell_3/BiasAdd_3BiasAdd8sequential_4/lstm_2/while/lstm_cell_3/MatMul_3:product:06sequential_4/lstm_2/while/lstm_cell_3/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@É
+sequential_4/lstm_2/while/lstm_cell_3/mul_4Mul'sequential_4_lstm_2_while_placeholder_2:sequential_4/lstm_2/while/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@É
+sequential_4/lstm_2/while/lstm_cell_3/mul_5Mul'sequential_4_lstm_2_while_placeholder_2:sequential_4/lstm_2/while/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@É
+sequential_4/lstm_2/while/lstm_cell_3/mul_6Mul'sequential_4_lstm_2_while_placeholder_2:sequential_4/lstm_2/while/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@É
+sequential_4/lstm_2/while/lstm_cell_3/mul_7Mul'sequential_4_lstm_2_while_placeholder_2:sequential_4/lstm_2/while/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@µ
4sequential_4/lstm_2/while/lstm_cell_3/ReadVariableOpReadVariableOp?sequential_4_lstm_2_while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0
9sequential_4/lstm_2/while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
;sequential_4/lstm_2/while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   
;sequential_4/lstm_2/while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
3sequential_4/lstm_2/while/lstm_cell_3/strided_sliceStridedSlice<sequential_4/lstm_2/while/lstm_cell_3/ReadVariableOp:value:0Bsequential_4/lstm_2/while/lstm_cell_3/strided_slice/stack:output:0Dsequential_4/lstm_2/while/lstm_cell_3/strided_slice/stack_1:output:0Dsequential_4/lstm_2/while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskÙ
.sequential_4/lstm_2/while/lstm_cell_3/MatMul_4MatMul/sequential_4/lstm_2/while/lstm_cell_3/mul_4:z:0<sequential_4/lstm_2/while/lstm_cell_3/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ö
)sequential_4/lstm_2/while/lstm_cell_3/addAddV26sequential_4/lstm_2/while/lstm_cell_3/BiasAdd:output:08sequential_4/lstm_2/while/lstm_cell_3/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
-sequential_4/lstm_2/while/lstm_cell_3/SigmoidSigmoid-sequential_4/lstm_2/while/lstm_cell_3/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@·
6sequential_4/lstm_2/while/lstm_cell_3/ReadVariableOp_1ReadVariableOp?sequential_4_lstm_2_while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0
;sequential_4/lstm_2/while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   
=sequential_4/lstm_2/while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
=sequential_4/lstm_2/while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
5sequential_4/lstm_2/while/lstm_cell_3/strided_slice_1StridedSlice>sequential_4/lstm_2/while/lstm_cell_3/ReadVariableOp_1:value:0Dsequential_4/lstm_2/while/lstm_cell_3/strided_slice_1/stack:output:0Fsequential_4/lstm_2/while/lstm_cell_3/strided_slice_1/stack_1:output:0Fsequential_4/lstm_2/while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskÛ
.sequential_4/lstm_2/while/lstm_cell_3/MatMul_5MatMul/sequential_4/lstm_2/while/lstm_cell_3/mul_5:z:0>sequential_4/lstm_2/while/lstm_cell_3/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ú
+sequential_4/lstm_2/while/lstm_cell_3/add_1AddV28sequential_4/lstm_2/while/lstm_cell_3/BiasAdd_1:output:08sequential_4/lstm_2/while/lstm_cell_3/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
/sequential_4/lstm_2/while/lstm_cell_3/Sigmoid_1Sigmoid/sequential_4/lstm_2/while/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Â
+sequential_4/lstm_2/while/lstm_cell_3/mul_8Mul3sequential_4/lstm_2/while/lstm_cell_3/Sigmoid_1:y:0'sequential_4_lstm_2_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@·
6sequential_4/lstm_2/while/lstm_cell_3/ReadVariableOp_2ReadVariableOp?sequential_4_lstm_2_while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0
;sequential_4/lstm_2/while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
=sequential_4/lstm_2/while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   
=sequential_4/lstm_2/while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
5sequential_4/lstm_2/while/lstm_cell_3/strided_slice_2StridedSlice>sequential_4/lstm_2/while/lstm_cell_3/ReadVariableOp_2:value:0Dsequential_4/lstm_2/while/lstm_cell_3/strided_slice_2/stack:output:0Fsequential_4/lstm_2/while/lstm_cell_3/strided_slice_2/stack_1:output:0Fsequential_4/lstm_2/while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskÛ
.sequential_4/lstm_2/while/lstm_cell_3/MatMul_6MatMul/sequential_4/lstm_2/while/lstm_cell_3/mul_6:z:0>sequential_4/lstm_2/while/lstm_cell_3/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ú
+sequential_4/lstm_2/while/lstm_cell_3/add_2AddV28sequential_4/lstm_2/while/lstm_cell_3/BiasAdd_2:output:08sequential_4/lstm_2/while/lstm_cell_3/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*sequential_4/lstm_2/while/lstm_cell_3/TanhTanh/sequential_4/lstm_2/while/lstm_cell_3/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ç
+sequential_4/lstm_2/while/lstm_cell_3/mul_9Mul1sequential_4/lstm_2/while/lstm_cell_3/Sigmoid:y:0.sequential_4/lstm_2/while/lstm_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@È
+sequential_4/lstm_2/while/lstm_cell_3/add_3AddV2/sequential_4/lstm_2/while/lstm_cell_3/mul_8:z:0/sequential_4/lstm_2/while/lstm_cell_3/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@·
6sequential_4/lstm_2/while/lstm_cell_3/ReadVariableOp_3ReadVariableOp?sequential_4_lstm_2_while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0
;sequential_4/lstm_2/while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   
=sequential_4/lstm_2/while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
=sequential_4/lstm_2/while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
5sequential_4/lstm_2/while/lstm_cell_3/strided_slice_3StridedSlice>sequential_4/lstm_2/while/lstm_cell_3/ReadVariableOp_3:value:0Dsequential_4/lstm_2/while/lstm_cell_3/strided_slice_3/stack:output:0Fsequential_4/lstm_2/while/lstm_cell_3/strided_slice_3/stack_1:output:0Fsequential_4/lstm_2/while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskÛ
.sequential_4/lstm_2/while/lstm_cell_3/MatMul_7MatMul/sequential_4/lstm_2/while/lstm_cell_3/mul_7:z:0>sequential_4/lstm_2/while/lstm_cell_3/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ú
+sequential_4/lstm_2/while/lstm_cell_3/add_4AddV28sequential_4/lstm_2/while/lstm_cell_3/BiasAdd_3:output:08sequential_4/lstm_2/while/lstm_cell_3/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
/sequential_4/lstm_2/while/lstm_cell_3/Sigmoid_2Sigmoid/sequential_4/lstm_2/while/lstm_cell_3/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,sequential_4/lstm_2/while/lstm_cell_3/Tanh_1Tanh/sequential_4/lstm_2/while/lstm_cell_3/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ì
,sequential_4/lstm_2/while/lstm_cell_3/mul_10Mul3sequential_4/lstm_2/while/lstm_cell_3/Sigmoid_2:y:00sequential_4/lstm_2/while/lstm_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
>sequential_4/lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_4_lstm_2_while_placeholder_1%sequential_4_lstm_2_while_placeholder0sequential_4/lstm_2/while/lstm_cell_3/mul_10:z:0*
_output_shapes
: *
element_dtype0:éèÒa
sequential_4/lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_4/lstm_2/while/addAddV2%sequential_4_lstm_2_while_placeholder(sequential_4/lstm_2/while/add/y:output:0*
T0*
_output_shapes
: c
!sequential_4/lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :·
sequential_4/lstm_2/while/add_1AddV2@sequential_4_lstm_2_while_sequential_4_lstm_2_while_loop_counter*sequential_4/lstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: 
"sequential_4/lstm_2/while/IdentityIdentity#sequential_4/lstm_2/while/add_1:z:0^sequential_4/lstm_2/while/NoOp*
T0*
_output_shapes
: º
$sequential_4/lstm_2/while/Identity_1IdentityFsequential_4_lstm_2_while_sequential_4_lstm_2_while_maximum_iterations^sequential_4/lstm_2/while/NoOp*
T0*
_output_shapes
: 
$sequential_4/lstm_2/while/Identity_2Identity!sequential_4/lstm_2/while/add:z:0^sequential_4/lstm_2/while/NoOp*
T0*
_output_shapes
: Õ
$sequential_4/lstm_2/while/Identity_3IdentityNsequential_4/lstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_4/lstm_2/while/NoOp*
T0*
_output_shapes
: :éèÒµ
$sequential_4/lstm_2/while/Identity_4Identity0sequential_4/lstm_2/while/lstm_cell_3/mul_10:z:0^sequential_4/lstm_2/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@´
$sequential_4/lstm_2/while/Identity_5Identity/sequential_4/lstm_2/while/lstm_cell_3/add_3:z:0^sequential_4/lstm_2/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¾
sequential_4/lstm_2/while/NoOpNoOp5^sequential_4/lstm_2/while/lstm_cell_3/ReadVariableOp7^sequential_4/lstm_2/while/lstm_cell_3/ReadVariableOp_17^sequential_4/lstm_2/while/lstm_cell_3/ReadVariableOp_27^sequential_4/lstm_2/while/lstm_cell_3/ReadVariableOp_3;^sequential_4/lstm_2/while/lstm_cell_3/split/ReadVariableOp=^sequential_4/lstm_2/while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Q
"sequential_4_lstm_2_while_identity+sequential_4/lstm_2/while/Identity:output:0"U
$sequential_4_lstm_2_while_identity_1-sequential_4/lstm_2/while/Identity_1:output:0"U
$sequential_4_lstm_2_while_identity_2-sequential_4/lstm_2/while/Identity_2:output:0"U
$sequential_4_lstm_2_while_identity_3-sequential_4/lstm_2/while/Identity_3:output:0"U
$sequential_4_lstm_2_while_identity_4-sequential_4/lstm_2/while/Identity_4:output:0"U
$sequential_4_lstm_2_while_identity_5-sequential_4/lstm_2/while/Identity_5:output:0"
=sequential_4_lstm_2_while_lstm_cell_3_readvariableop_resource?sequential_4_lstm_2_while_lstm_cell_3_readvariableop_resource_0"
Esequential_4_lstm_2_while_lstm_cell_3_split_1_readvariableop_resourceGsequential_4_lstm_2_while_lstm_cell_3_split_1_readvariableop_resource_0"
Csequential_4_lstm_2_while_lstm_cell_3_split_readvariableop_resourceEsequential_4_lstm_2_while_lstm_cell_3_split_readvariableop_resource_0"
=sequential_4_lstm_2_while_sequential_4_lstm_2_strided_slice_1?sequential_4_lstm_2_while_sequential_4_lstm_2_strided_slice_1_0"ø
ysequential_4_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_2_tensorarrayunstack_tensorlistfromtensor{sequential_4_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : 2l
4sequential_4/lstm_2/while/lstm_cell_3/ReadVariableOp4sequential_4/lstm_2/while/lstm_cell_3/ReadVariableOp2p
6sequential_4/lstm_2/while/lstm_cell_3/ReadVariableOp_16sequential_4/lstm_2/while/lstm_cell_3/ReadVariableOp_12p
6sequential_4/lstm_2/while/lstm_cell_3/ReadVariableOp_26sequential_4/lstm_2/while/lstm_cell_3/ReadVariableOp_22p
6sequential_4/lstm_2/while/lstm_cell_3/ReadVariableOp_36sequential_4/lstm_2/while/lstm_cell_3/ReadVariableOp_32x
:sequential_4/lstm_2/while/lstm_cell_3/split/ReadVariableOp:sequential_4/lstm_2/while/lstm_cell_3/split/ReadVariableOp2|
<sequential_4/lstm_2/while/lstm_cell_3/split_1/ReadVariableOp<sequential_4/lstm_2/while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ñ7
ü
A__inference_lstm_2_layer_call_and_return_conditional_losses_21949

inputs$
lstm_cell_3_21867:	d 
lstm_cell_3_21869:	$
lstm_cell_3_21871:	@
identity¢#lstm_cell_3/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskì
#lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_3_21867lstm_cell_3_21869lstm_cell_3_21871*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_21866n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¯
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_3_21867lstm_cell_3_21869lstm_cell_3_21871*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_21880*
condR
while_cond_21879*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
NoOpNoOp$^lstm_cell_3/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: : : 2J
#lstm_cell_3/StatefulPartitionedCall#lstm_cell_3/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs


ó
B__inference_dense_4_layer_call_and_return_conditional_losses_22545

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
á
Î
$sequential_4_lstm_2_while_cond_21600D
@sequential_4_lstm_2_while_sequential_4_lstm_2_while_loop_counterJ
Fsequential_4_lstm_2_while_sequential_4_lstm_2_while_maximum_iterations)
%sequential_4_lstm_2_while_placeholder+
'sequential_4_lstm_2_while_placeholder_1+
'sequential_4_lstm_2_while_placeholder_2+
'sequential_4_lstm_2_while_placeholder_3F
Bsequential_4_lstm_2_while_less_sequential_4_lstm_2_strided_slice_1[
Wsequential_4_lstm_2_while_sequential_4_lstm_2_while_cond_21600___redundant_placeholder0[
Wsequential_4_lstm_2_while_sequential_4_lstm_2_while_cond_21600___redundant_placeholder1[
Wsequential_4_lstm_2_while_sequential_4_lstm_2_while_cond_21600___redundant_placeholder2[
Wsequential_4_lstm_2_while_sequential_4_lstm_2_while_cond_21600___redundant_placeholder3&
"sequential_4_lstm_2_while_identity
²
sequential_4/lstm_2/while/LessLess%sequential_4_lstm_2_while_placeholderBsequential_4_lstm_2_while_less_sequential_4_lstm_2_strided_slice_1*
T0*
_output_shapes
: s
"sequential_4/lstm_2/while/IdentityIdentity"sequential_4/lstm_2/while/Less:z:0*
T0
*
_output_shapes
: "Q
"sequential_4_lstm_2_while_identity+sequential_4/lstm_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
°§
â
G__inference_sequential_4_layer_call_and_return_conditional_losses_23455

inputs5
"embedding_4_embedding_lookup_23196:	ÄdC
0lstm_2_lstm_cell_3_split_readvariableop_resource:	dA
2lstm_2_lstm_cell_3_split_1_readvariableop_resource:	=
*lstm_2_lstm_cell_3_readvariableop_resource:	@8
&dense_4_matmul_readvariableop_resource:@ 5
'dense_4_biasadd_readvariableop_resource: 8
&dense_5_matmul_readvariableop_resource: 5
'dense_5_biasadd_readvariableop_resource:
identity¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢dense_5/BiasAdd/ReadVariableOp¢dense_5/MatMul/ReadVariableOp¢embedding_4/embedding_lookup¢!lstm_2/lstm_cell_3/ReadVariableOp¢#lstm_2/lstm_cell_3/ReadVariableOp_1¢#lstm_2/lstm_cell_3/ReadVariableOp_2¢#lstm_2/lstm_cell_3/ReadVariableOp_3¢'lstm_2/lstm_cell_3/split/ReadVariableOp¢)lstm_2/lstm_cell_3/split_1/ReadVariableOp¢lstm_2/whileb
embedding_4/CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿúê
embedding_4/embedding_lookupResourceGather"embedding_4_embedding_lookup_23196embedding_4/Cast:y:0*
Tindices0*5
_class+
)'loc:@embedding_4/embedding_lookup/23196*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúd*
dtype0Æ
%embedding_4/embedding_lookup/IdentityIdentity%embedding_4/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_4/embedding_lookup/23196*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúd
'embedding_4/embedding_lookup/Identity_1Identity.embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúdl
lstm_2/ShapeShape0embedding_4/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:d
lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
lstm_2/strided_sliceStridedSlicelstm_2/Shape:output:0#lstm_2/strided_slice/stack:output:0%lstm_2/strided_slice/stack_1:output:0%lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@
lstm_2/zeros/packedPacklstm_2/strided_slice:output:0lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_2/zerosFilllstm_2/zeros/packed:output:0lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@
lstm_2/zeros_1/packedPacklstm_2/strided_slice:output:0 lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_2/zeros_1Filllstm_2/zeros_1/packed:output:0lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@j
lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¦
lstm_2/transpose	Transpose0embedding_4/embedding_lookup/Identity_1:output:0lstm_2/transpose/perm:output:0*
T0*,
_output_shapes
:úÿÿÿÿÿÿÿÿÿdR
lstm_2/Shape_1Shapelstm_2/transpose:y:0*
T0*
_output_shapes
:f
lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
lstm_2/strided_slice_1StridedSlicelstm_2/Shape_1:output:0%lstm_2/strided_slice_1/stack:output:0'lstm_2/strided_slice_1/stack_1:output:0'lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÉ
lstm_2/TensorArrayV2TensorListReserve+lstm_2/TensorArrayV2/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   õ
.lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_2/transpose:y:0Elstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒf
lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_2/strided_slice_2StridedSlicelstm_2/transpose:y:0%lstm_2/strided_slice_2/stack:output:0'lstm_2/strided_slice_2/stack_1:output:0'lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskq
"lstm_2/lstm_cell_3/ones_like/ShapeShapelstm_2/strided_slice_2:output:0*
T0*
_output_shapes
:g
"lstm_2/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
lstm_2/lstm_cell_3/ones_likeFill+lstm_2/lstm_cell_3/ones_like/Shape:output:0+lstm_2/lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
$lstm_2/lstm_cell_3/ones_like_1/ShapeShapelstm_2/zeros:output:0*
T0*
_output_shapes
:i
$lstm_2/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
lstm_2/lstm_cell_3/ones_like_1Fill-lstm_2/lstm_cell_3/ones_like_1/Shape:output:0-lstm_2/lstm_cell_3/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/lstm_cell_3/mulMullstm_2/strided_slice_2:output:0%lstm_2/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_2/lstm_cell_3/mul_1Mullstm_2/strided_slice_2:output:0%lstm_2/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_2/lstm_cell_3/mul_2Mullstm_2/strided_slice_2:output:0%lstm_2/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_2/lstm_cell_3/mul_3Mullstm_2/strided_slice_2:output:0%lstm_2/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
"lstm_2/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
'lstm_2/lstm_cell_3/split/ReadVariableOpReadVariableOp0lstm_2_lstm_cell_3_split_readvariableop_resource*
_output_shapes
:	d*
dtype0×
lstm_2/lstm_cell_3/splitSplit+lstm_2/lstm_cell_3/split/split_dim:output:0/lstm_2/lstm_cell_3/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:d@:d@:d@:d@*
	num_split
lstm_2/lstm_cell_3/MatMulMatMullstm_2/lstm_cell_3/mul:z:0!lstm_2/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/lstm_cell_3/MatMul_1MatMullstm_2/lstm_cell_3/mul_1:z:0!lstm_2/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/lstm_cell_3/MatMul_2MatMullstm_2/lstm_cell_3/mul_2:z:0!lstm_2/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/lstm_cell_3/MatMul_3MatMullstm_2/lstm_cell_3/mul_3:z:0!lstm_2/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@f
$lstm_2/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
)lstm_2/lstm_cell_3/split_1/ReadVariableOpReadVariableOp2lstm_2_lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0Í
lstm_2/lstm_cell_3/split_1Split-lstm_2/lstm_cell_3/split_1/split_dim:output:01lstm_2/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split¡
lstm_2/lstm_cell_3/BiasAddBiasAdd#lstm_2/lstm_cell_3/MatMul:product:0#lstm_2/lstm_cell_3/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
lstm_2/lstm_cell_3/BiasAdd_1BiasAdd%lstm_2/lstm_cell_3/MatMul_1:product:0#lstm_2/lstm_cell_3/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
lstm_2/lstm_cell_3/BiasAdd_2BiasAdd%lstm_2/lstm_cell_3/MatMul_2:product:0#lstm_2/lstm_cell_3/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
lstm_2/lstm_cell_3/BiasAdd_3BiasAdd%lstm_2/lstm_cell_3/MatMul_3:product:0#lstm_2/lstm_cell_3/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/lstm_cell_3/mul_4Mullstm_2/zeros:output:0'lstm_2/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/lstm_cell_3/mul_5Mullstm_2/zeros:output:0'lstm_2/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/lstm_cell_3/mul_6Mullstm_2/zeros:output:0'lstm_2/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/lstm_cell_3/mul_7Mullstm_2/zeros:output:0'lstm_2/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
!lstm_2/lstm_cell_3/ReadVariableOpReadVariableOp*lstm_2_lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0w
&lstm_2/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(lstm_2/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   y
(lstm_2/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ê
 lstm_2/lstm_cell_3/strided_sliceStridedSlice)lstm_2/lstm_cell_3/ReadVariableOp:value:0/lstm_2/lstm_cell_3/strided_slice/stack:output:01lstm_2/lstm_cell_3/strided_slice/stack_1:output:01lstm_2/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask 
lstm_2/lstm_cell_3/MatMul_4MatMullstm_2/lstm_cell_3/mul_4:z:0)lstm_2/lstm_cell_3/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/lstm_cell_3/addAddV2#lstm_2/lstm_cell_3/BiasAdd:output:0%lstm_2/lstm_cell_3/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
lstm_2/lstm_cell_3/SigmoidSigmoidlstm_2/lstm_cell_3/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
#lstm_2/lstm_cell_3/ReadVariableOp_1ReadVariableOp*lstm_2_lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0y
(lstm_2/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   {
*lstm_2/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_2/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
"lstm_2/lstm_cell_3/strided_slice_1StridedSlice+lstm_2/lstm_cell_3/ReadVariableOp_1:value:01lstm_2/lstm_cell_3/strided_slice_1/stack:output:03lstm_2/lstm_cell_3/strided_slice_1/stack_1:output:03lstm_2/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask¢
lstm_2/lstm_cell_3/MatMul_5MatMullstm_2/lstm_cell_3/mul_5:z:0+lstm_2/lstm_cell_3/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡
lstm_2/lstm_cell_3/add_1AddV2%lstm_2/lstm_cell_3/BiasAdd_1:output:0%lstm_2/lstm_cell_3/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
lstm_2/lstm_cell_3/Sigmoid_1Sigmoidlstm_2/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/lstm_cell_3/mul_8Mul lstm_2/lstm_cell_3/Sigmoid_1:y:0lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
#lstm_2/lstm_cell_3/ReadVariableOp_2ReadVariableOp*lstm_2_lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0y
(lstm_2/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_2/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   {
*lstm_2/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
"lstm_2/lstm_cell_3/strided_slice_2StridedSlice+lstm_2/lstm_cell_3/ReadVariableOp_2:value:01lstm_2/lstm_cell_3/strided_slice_2/stack:output:03lstm_2/lstm_cell_3/strided_slice_2/stack_1:output:03lstm_2/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask¢
lstm_2/lstm_cell_3/MatMul_6MatMullstm_2/lstm_cell_3/mul_6:z:0+lstm_2/lstm_cell_3/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡
lstm_2/lstm_cell_3/add_2AddV2%lstm_2/lstm_cell_3/BiasAdd_2:output:0%lstm_2/lstm_cell_3/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
lstm_2/lstm_cell_3/TanhTanhlstm_2/lstm_cell_3/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/lstm_cell_3/mul_9Mullstm_2/lstm_cell_3/Sigmoid:y:0lstm_2/lstm_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/lstm_cell_3/add_3AddV2lstm_2/lstm_cell_3/mul_8:z:0lstm_2/lstm_cell_3/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
#lstm_2/lstm_cell_3/ReadVariableOp_3ReadVariableOp*lstm_2_lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0y
(lstm_2/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   {
*lstm_2/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_2/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
"lstm_2/lstm_cell_3/strided_slice_3StridedSlice+lstm_2/lstm_cell_3/ReadVariableOp_3:value:01lstm_2/lstm_cell_3/strided_slice_3/stack:output:03lstm_2/lstm_cell_3/strided_slice_3/stack_1:output:03lstm_2/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask¢
lstm_2/lstm_cell_3/MatMul_7MatMullstm_2/lstm_cell_3/mul_7:z:0+lstm_2/lstm_cell_3/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡
lstm_2/lstm_cell_3/add_4AddV2%lstm_2/lstm_cell_3/BiasAdd_3:output:0%lstm_2/lstm_cell_3/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
lstm_2/lstm_cell_3/Sigmoid_2Sigmoidlstm_2/lstm_cell_3/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
lstm_2/lstm_cell_3/Tanh_1Tanhlstm_2/lstm_cell_3/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/lstm_cell_3/mul_10Mul lstm_2/lstm_cell_3/Sigmoid_2:y:0lstm_2/lstm_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@u
$lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Í
lstm_2/TensorArrayV2_1TensorListReserve-lstm_2/TensorArrayV2_1/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒM
lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Õ
lstm_2/whileWhile"lstm_2/while/loop_counter:output:0(lstm_2/while/maximum_iterations:output:0lstm_2/time:output:0lstm_2/TensorArrayV2_1:handle:0lstm_2/zeros:output:0lstm_2/zeros_1:output:0lstm_2/strided_slice_1:output:0>lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:00lstm_2_lstm_cell_3_split_readvariableop_resource2lstm_2_lstm_cell_3_split_1_readvariableop_resource*lstm_2_lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstm_2_while_body_23307*#
condR
lstm_2_while_cond_23306*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
7lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ø
)lstm_2/TensorArrayV2Stack/TensorListStackTensorListStacklstm_2/while:output:3@lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:úÿÿÿÿÿÿÿÿÿ@*
element_dtype0o
lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿh
lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ª
lstm_2/strided_slice_3StridedSlice2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_2/strided_slice_3/stack:output:0'lstm_2/strided_slice_3/stack_1:output:0'lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maskl
lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¬
lstm_2/transpose_1	Transpose2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_2/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿú@b
lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
dense_4/MatMulMatMullstm_2/strided_slice_3:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitydense_5/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^embedding_4/embedding_lookup"^lstm_2/lstm_cell_3/ReadVariableOp$^lstm_2/lstm_cell_3/ReadVariableOp_1$^lstm_2/lstm_cell_3/ReadVariableOp_2$^lstm_2/lstm_cell_3/ReadVariableOp_3(^lstm_2/lstm_cell_3/split/ReadVariableOp*^lstm_2/lstm_cell_3/split_1/ReadVariableOp^lstm_2/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿú: : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2<
embedding_4/embedding_lookupembedding_4/embedding_lookup2F
!lstm_2/lstm_cell_3/ReadVariableOp!lstm_2/lstm_cell_3/ReadVariableOp2J
#lstm_2/lstm_cell_3/ReadVariableOp_1#lstm_2/lstm_cell_3/ReadVariableOp_12J
#lstm_2/lstm_cell_3/ReadVariableOp_2#lstm_2/lstm_cell_3/ReadVariableOp_22J
#lstm_2/lstm_cell_3/ReadVariableOp_3#lstm_2/lstm_cell_3/ReadVariableOp_32R
'lstm_2/lstm_cell_3/split/ReadVariableOp'lstm_2/lstm_cell_3/split/ReadVariableOp2V
)lstm_2/lstm_cell_3/split_1/ReadVariableOp)lstm_2/lstm_cell_3/split_1/ReadVariableOp2
lstm_2/whilelstm_2/while:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs
¿t
	
while_body_24653
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_3_split_readvariableop_resource_0:	dB
3while_lstm_cell_3_split_1_readvariableop_resource_0:	>
+while_lstm_cell_3_readvariableop_resource_0:	@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_3_split_readvariableop_resource:	d@
1while_lstm_cell_3_split_1_readvariableop_resource:	<
)while_lstm_cell_3_readvariableop_resource:	@¢ while/lstm_cell_3/ReadVariableOp¢"while/lstm_cell_3/ReadVariableOp_1¢"while/lstm_cell_3/ReadVariableOp_2¢"while/lstm_cell_3/ReadVariableOp_3¢&while/lstm_cell_3/split/ReadVariableOp¢(while/lstm_cell_3/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
!while/lstm_cell_3/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?­
while/lstm_cell_3/ones_likeFill*while/lstm_cell_3/ones_like/Shape:output:0*while/lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
#while/lstm_cell_3/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:h
#while/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
while/lstm_cell_3/ones_like_1Fill,while/lstm_cell_3/ones_like_1/Shape:output:0,while/lstm_cell_3/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
while/lstm_cell_3/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¨
while/lstm_cell_3/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¨
while/lstm_cell_3/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¨
while/lstm_cell_3/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
&while/lstm_cell_3/split/ReadVariableOpReadVariableOp1while_lstm_cell_3_split_readvariableop_resource_0*
_output_shapes
:	d*
dtype0Ô
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0.while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:d@:d@:d@:d@*
	num_split
while/lstm_cell_3/MatMulMatMulwhile/lstm_cell_3/mul:z:0 while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/MatMul_1MatMulwhile/lstm_cell_3/mul_1:z:0 while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/MatMul_2MatMulwhile/lstm_cell_3/mul_2:z:0 while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/MatMul_3MatMulwhile/lstm_cell_3/mul_3:z:0 while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
#while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
(while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Ê
while/lstm_cell_3/split_1Split,while/lstm_cell_3/split_1/split_dim:output:00while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split
while/lstm_cell_3/BiasAddBiasAdd"while/lstm_cell_3/MatMul:product:0"while/lstm_cell_3/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
while/lstm_cell_3/BiasAdd_1BiasAdd$while/lstm_cell_3/MatMul_1:product:0"while/lstm_cell_3/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
while/lstm_cell_3/BiasAdd_2BiasAdd$while/lstm_cell_3/MatMul_2:product:0"while/lstm_cell_3/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
while/lstm_cell_3/BiasAdd_3BiasAdd$while/lstm_cell_3/MatMul_3:product:0"while/lstm_cell_3/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_4Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_5Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_6Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_7Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 while/lstm_cell_3/ReadVariableOpReadVariableOp+while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0v
%while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   x
'while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Å
while/lstm_cell_3/strided_sliceStridedSlice(while/lstm_cell_3/ReadVariableOp:value:0.while/lstm_cell_3/strided_slice/stack:output:00while/lstm_cell_3/strided_slice/stack_1:output:00while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_4MatMulwhile/lstm_cell_3/mul_4:z:0(while/lstm_cell_3/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/addAddV2"while/lstm_cell_3/BiasAdd:output:0$while/lstm_cell_3/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
while/lstm_cell_3/SigmoidSigmoidwhile/lstm_cell_3/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"while/lstm_cell_3/ReadVariableOp_1ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0x
'while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   z
)while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ï
!while/lstm_cell_3/strided_slice_1StridedSlice*while/lstm_cell_3/ReadVariableOp_1:value:00while/lstm_cell_3/strided_slice_1/stack:output:02while/lstm_cell_3/strided_slice_1/stack_1:output:02while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_5MatMulwhile/lstm_cell_3/mul_5:z:0*while/lstm_cell_3/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/add_1AddV2$while/lstm_cell_3/BiasAdd_1:output:0$while/lstm_cell_3/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@u
while/lstm_cell_3/Sigmoid_1Sigmoidwhile/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_8Mulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"while/lstm_cell_3/ReadVariableOp_2ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0x
'while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   z
)while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ï
!while/lstm_cell_3/strided_slice_2StridedSlice*while/lstm_cell_3/ReadVariableOp_2:value:00while/lstm_cell_3/strided_slice_2/stack:output:02while/lstm_cell_3/strided_slice_2/stack_1:output:02while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_6MatMulwhile/lstm_cell_3/mul_6:z:0*while/lstm_cell_3/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/add_2AddV2$while/lstm_cell_3/BiasAdd_2:output:0$while/lstm_cell_3/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
while/lstm_cell_3/TanhTanhwhile/lstm_cell_3/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_9Mulwhile/lstm_cell_3/Sigmoid:y:0while/lstm_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/add_3AddV2while/lstm_cell_3/mul_8:z:0while/lstm_cell_3/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"while/lstm_cell_3/ReadVariableOp_3ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0x
'while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   z
)while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ï
!while/lstm_cell_3/strided_slice_3StridedSlice*while/lstm_cell_3/ReadVariableOp_3:value:00while/lstm_cell_3/strided_slice_3/stack:output:02while/lstm_cell_3/strided_slice_3/stack_1:output:02while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_7MatMulwhile/lstm_cell_3/mul_7:z:0*while/lstm_cell_3/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/add_4AddV2$while/lstm_cell_3/BiasAdd_3:output:0$while/lstm_cell_3/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@u
while/lstm_cell_3/Sigmoid_2Sigmoidwhile/lstm_cell_3/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
while/lstm_cell_3/Tanh_1Tanhwhile/lstm_cell_3/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_10Mulwhile/lstm_cell_3/Sigmoid_2:y:0while/lstm_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_10:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒy
while/Identity_4Identitywhile/lstm_cell_3/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
while/Identity_5Identitywhile/lstm_cell_3/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@²

while/NoOpNoOp!^while/lstm_cell_3/ReadVariableOp#^while/lstm_cell_3/ReadVariableOp_1#^while/lstm_cell_3/ReadVariableOp_2#^while/lstm_cell_3/ReadVariableOp_3'^while/lstm_cell_3/split/ReadVariableOp)^while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_3_readvariableop_resource+while_lstm_cell_3_readvariableop_resource_0"h
1while_lstm_cell_3_split_1_readvariableop_resource3while_lstm_cell_3_split_1_readvariableop_resource_0"d
/while_lstm_cell_3_split_readvariableop_resource1while_lstm_cell_3_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : 2D
 while/lstm_cell_3/ReadVariableOp while/lstm_cell_3/ReadVariableOp2H
"while/lstm_cell_3/ReadVariableOp_1"while/lstm_cell_3/ReadVariableOp_12H
"while/lstm_cell_3/ReadVariableOp_2"while/lstm_cell_3/ReadVariableOp_22H
"while/lstm_cell_3/ReadVariableOp_3"while/lstm_cell_3/ReadVariableOp_32P
&while/lstm_cell_3/split/ReadVariableOp&while/lstm_cell_3/split/ReadVariableOp2T
(while/lstm_cell_3/split_1/ReadVariableOp(while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
Î	
Ã
,__inference_sequential_4_layer_call_fn_23192

inputs
unknown:	Äd
	unknown_0:	d
	unknown_1:	
	unknown_2:	@
	unknown_3:@ 
	unknown_4: 
	unknown_5: 
	unknown_6:
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_23056o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿú: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs
°
¾
while_cond_21879
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_21879___redundant_placeholder03
/while_while_cond_21879___redundant_placeholder13
/while_while_cond_21879___redundant_placeholder23
/while_while_cond_21879___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
¿t
	
while_body_24039
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_3_split_readvariableop_resource_0:	dB
3while_lstm_cell_3_split_1_readvariableop_resource_0:	>
+while_lstm_cell_3_readvariableop_resource_0:	@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_3_split_readvariableop_resource:	d@
1while_lstm_cell_3_split_1_readvariableop_resource:	<
)while_lstm_cell_3_readvariableop_resource:	@¢ while/lstm_cell_3/ReadVariableOp¢"while/lstm_cell_3/ReadVariableOp_1¢"while/lstm_cell_3/ReadVariableOp_2¢"while/lstm_cell_3/ReadVariableOp_3¢&while/lstm_cell_3/split/ReadVariableOp¢(while/lstm_cell_3/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
!while/lstm_cell_3/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?­
while/lstm_cell_3/ones_likeFill*while/lstm_cell_3/ones_like/Shape:output:0*while/lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
#while/lstm_cell_3/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:h
#while/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
while/lstm_cell_3/ones_like_1Fill,while/lstm_cell_3/ones_like_1/Shape:output:0,while/lstm_cell_3/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
while/lstm_cell_3/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¨
while/lstm_cell_3/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¨
while/lstm_cell_3/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¨
while/lstm_cell_3/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
&while/lstm_cell_3/split/ReadVariableOpReadVariableOp1while_lstm_cell_3_split_readvariableop_resource_0*
_output_shapes
:	d*
dtype0Ô
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0.while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:d@:d@:d@:d@*
	num_split
while/lstm_cell_3/MatMulMatMulwhile/lstm_cell_3/mul:z:0 while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/MatMul_1MatMulwhile/lstm_cell_3/mul_1:z:0 while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/MatMul_2MatMulwhile/lstm_cell_3/mul_2:z:0 while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/MatMul_3MatMulwhile/lstm_cell_3/mul_3:z:0 while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
#while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
(while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Ê
while/lstm_cell_3/split_1Split,while/lstm_cell_3/split_1/split_dim:output:00while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split
while/lstm_cell_3/BiasAddBiasAdd"while/lstm_cell_3/MatMul:product:0"while/lstm_cell_3/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
while/lstm_cell_3/BiasAdd_1BiasAdd$while/lstm_cell_3/MatMul_1:product:0"while/lstm_cell_3/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
while/lstm_cell_3/BiasAdd_2BiasAdd$while/lstm_cell_3/MatMul_2:product:0"while/lstm_cell_3/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
while/lstm_cell_3/BiasAdd_3BiasAdd$while/lstm_cell_3/MatMul_3:product:0"while/lstm_cell_3/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_4Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_5Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_6Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_7Mulwhile_placeholder_2&while/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 while/lstm_cell_3/ReadVariableOpReadVariableOp+while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0v
%while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   x
'while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Å
while/lstm_cell_3/strided_sliceStridedSlice(while/lstm_cell_3/ReadVariableOp:value:0.while/lstm_cell_3/strided_slice/stack:output:00while/lstm_cell_3/strided_slice/stack_1:output:00while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_4MatMulwhile/lstm_cell_3/mul_4:z:0(while/lstm_cell_3/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/addAddV2"while/lstm_cell_3/BiasAdd:output:0$while/lstm_cell_3/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
while/lstm_cell_3/SigmoidSigmoidwhile/lstm_cell_3/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"while/lstm_cell_3/ReadVariableOp_1ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0x
'while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   z
)while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ï
!while/lstm_cell_3/strided_slice_1StridedSlice*while/lstm_cell_3/ReadVariableOp_1:value:00while/lstm_cell_3/strided_slice_1/stack:output:02while/lstm_cell_3/strided_slice_1/stack_1:output:02while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_5MatMulwhile/lstm_cell_3/mul_5:z:0*while/lstm_cell_3/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/add_1AddV2$while/lstm_cell_3/BiasAdd_1:output:0$while/lstm_cell_3/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@u
while/lstm_cell_3/Sigmoid_1Sigmoidwhile/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_8Mulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"while/lstm_cell_3/ReadVariableOp_2ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0x
'while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   z
)while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ï
!while/lstm_cell_3/strided_slice_2StridedSlice*while/lstm_cell_3/ReadVariableOp_2:value:00while/lstm_cell_3/strided_slice_2/stack:output:02while/lstm_cell_3/strided_slice_2/stack_1:output:02while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_6MatMulwhile/lstm_cell_3/mul_6:z:0*while/lstm_cell_3/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/add_2AddV2$while/lstm_cell_3/BiasAdd_2:output:0$while/lstm_cell_3/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
while/lstm_cell_3/TanhTanhwhile/lstm_cell_3/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_9Mulwhile/lstm_cell_3/Sigmoid:y:0while/lstm_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/add_3AddV2while/lstm_cell_3/mul_8:z:0while/lstm_cell_3/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"while/lstm_cell_3/ReadVariableOp_3ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0x
'while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   z
)while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ï
!while/lstm_cell_3/strided_slice_3StridedSlice*while/lstm_cell_3/ReadVariableOp_3:value:00while/lstm_cell_3/strided_slice_3/stack:output:02while/lstm_cell_3/strided_slice_3/stack_1:output:02while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_7MatMulwhile/lstm_cell_3/mul_7:z:0*while/lstm_cell_3/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/add_4AddV2$while/lstm_cell_3/BiasAdd_3:output:0$while/lstm_cell_3/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@u
while/lstm_cell_3/Sigmoid_2Sigmoidwhile/lstm_cell_3/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
while/lstm_cell_3/Tanh_1Tanhwhile/lstm_cell_3/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_10Mulwhile/lstm_cell_3/Sigmoid_2:y:0while/lstm_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_10:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒy
while/Identity_4Identitywhile/lstm_cell_3/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
while/Identity_5Identitywhile/lstm_cell_3/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@²

while/NoOpNoOp!^while/lstm_cell_3/ReadVariableOp#^while/lstm_cell_3/ReadVariableOp_1#^while/lstm_cell_3/ReadVariableOp_2#^while/lstm_cell_3/ReadVariableOp_3'^while/lstm_cell_3/split/ReadVariableOp)^while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_3_readvariableop_resource+while_lstm_cell_3_readvariableop_resource_0"h
1while_lstm_cell_3_split_1_readvariableop_resource3while_lstm_cell_3_split_1_readvariableop_resource_0"d
/while_lstm_cell_3_split_readvariableop_resource1while_lstm_cell_3_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : 2D
 while/lstm_cell_3/ReadVariableOp while/lstm_cell_3/ReadVariableOp2H
"while/lstm_cell_3/ReadVariableOp_1"while/lstm_cell_3/ReadVariableOp_12H
"while/lstm_cell_3/ReadVariableOp_2"while/lstm_cell_3/ReadVariableOp_22H
"while/lstm_cell_3/ReadVariableOp_3"while/lstm_cell_3/ReadVariableOp_32P
&while/lstm_cell_3/split/ReadVariableOp&while/lstm_cell_3/split/ReadVariableOp2T
(while/lstm_cell_3/split_1/ReadVariableOp(while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
°
¾
while_cond_22793
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_22793___redundant_placeholder03
/while_while_cond_22793___redundant_placeholder13
/while_while_cond_22793___redundant_placeholder23
/while_while_cond_22793___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
¿	
Å
#__inference_signature_wrapper_23869
embedding_4_input
unknown:	Äd
	unknown_0:	d
	unknown_1:	
	unknown_2:	@
	unknown_3:@ 
	unknown_4: 
	unknown_5: 
	unknown_6:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallembedding_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_21749o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿú: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
+
_user_specified_nameembedding_4_input


ó
B__inference_dense_5_layer_call_and_return_conditional_losses_22562

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ô	
Ê
lstm_2_while_cond_23306*
&lstm_2_while_lstm_2_while_loop_counter0
,lstm_2_while_lstm_2_while_maximum_iterations
lstm_2_while_placeholder
lstm_2_while_placeholder_1
lstm_2_while_placeholder_2
lstm_2_while_placeholder_3,
(lstm_2_while_less_lstm_2_strided_slice_1A
=lstm_2_while_lstm_2_while_cond_23306___redundant_placeholder0A
=lstm_2_while_lstm_2_while_cond_23306___redundant_placeholder1A
=lstm_2_while_lstm_2_while_cond_23306___redundant_placeholder2A
=lstm_2_while_lstm_2_while_cond_23306___redundant_placeholder3
lstm_2_while_identity
~
lstm_2/while/LessLesslstm_2_while_placeholder(lstm_2_while_less_lstm_2_strided_slice_1*
T0*
_output_shapes
: Y
lstm_2/while/IdentityIdentitylstm_2/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_2_while_identitylstm_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
Î
Þ
A__inference_lstm_2_layer_call_and_return_conditional_losses_24173
inputs_0<
)lstm_cell_3_split_readvariableop_resource:	d:
+lstm_cell_3_split_1_readvariableop_resource:	6
#lstm_cell_3_readvariableop_resource:	@
identity¢lstm_cell_3/ReadVariableOp¢lstm_cell_3/ReadVariableOp_1¢lstm_cell_3/ReadVariableOp_2¢lstm_cell_3/ReadVariableOp_3¢ lstm_cell_3/split/ReadVariableOp¢"lstm_cell_3/split_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskc
lstm_cell_3/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:`
lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_3/ones_likeFill$lstm_cell_3/ones_like/Shape:output:0$lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
lstm_cell_3/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:b
lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
lstm_cell_3/ones_like_1Fill&lstm_cell_3/ones_like_1/Shape:output:0&lstm_cell_3/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/mulMulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/mul_1Mulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/mul_2Mulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/mul_3Mulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_cell_3/split/ReadVariableOpReadVariableOp)lstm_cell_3_split_readvariableop_resource*
_output_shapes
:	d*
dtype0Â
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0(lstm_cell_3/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:d@:d@:d@:d@*
	num_split
lstm_cell_3/MatMulMatMullstm_cell_3/mul:z:0lstm_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/MatMul_1MatMullstm_cell_3/mul_1:z:0lstm_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/MatMul_2MatMullstm_cell_3/mul_2:z:0lstm_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/MatMul_3MatMullstm_cell_3/mul_3:z:0lstm_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
"lstm_cell_3/split_1/ReadVariableOpReadVariableOp+lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0¸
lstm_cell_3/split_1Split&lstm_cell_3/split_1/split_dim:output:0*lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split
lstm_cell_3/BiasAddBiasAddlstm_cell_3/MatMul:product:0lstm_cell_3/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/BiasAdd_1BiasAddlstm_cell_3/MatMul_1:product:0lstm_cell_3/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/BiasAdd_2BiasAddlstm_cell_3/MatMul_2:product:0lstm_cell_3/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/BiasAdd_3BiasAddlstm_cell_3/MatMul_3:product:0lstm_cell_3/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
lstm_cell_3/mul_4Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
lstm_cell_3/mul_5Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
lstm_cell_3/mul_6Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
lstm_cell_3/mul_7Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/ReadVariableOpReadVariableOp#lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0p
lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   r
!lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
lstm_cell_3/strided_sliceStridedSlice"lstm_cell_3/ReadVariableOp:value:0(lstm_cell_3/strided_slice/stack:output:0*lstm_cell_3/strided_slice/stack_1:output:0*lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
lstm_cell_3/MatMul_4MatMullstm_cell_3/mul_4:z:0"lstm_cell_3/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/addAddV2lstm_cell_3/BiasAdd:output:0lstm_cell_3/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
lstm_cell_3/SigmoidSigmoidlstm_cell_3/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/ReadVariableOp_1ReadVariableOp#lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0r
!lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   t
#lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ±
lstm_cell_3/strided_slice_1StridedSlice$lstm_cell_3/ReadVariableOp_1:value:0*lstm_cell_3/strided_slice_1/stack:output:0,lstm_cell_3/strided_slice_1/stack_1:output:0,lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
lstm_cell_3/MatMul_5MatMullstm_cell_3/mul_5:z:0$lstm_cell_3/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/add_1AddV2lstm_cell_3/BiasAdd_1:output:0lstm_cell_3/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
lstm_cell_3/mul_8Mullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/ReadVariableOp_2ReadVariableOp#lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0r
!lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   t
#lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ±
lstm_cell_3/strided_slice_2StridedSlice$lstm_cell_3/ReadVariableOp_2:value:0*lstm_cell_3/strided_slice_2/stack:output:0,lstm_cell_3/strided_slice_2/stack_1:output:0,lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
lstm_cell_3/MatMul_6MatMullstm_cell_3/mul_6:z:0$lstm_cell_3/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/add_2AddV2lstm_cell_3/BiasAdd_2:output:0lstm_cell_3/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
lstm_cell_3/TanhTanhlstm_cell_3/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
lstm_cell_3/mul_9Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
lstm_cell_3/add_3AddV2lstm_cell_3/mul_8:z:0lstm_cell_3/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/ReadVariableOp_3ReadVariableOp#lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0r
!lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   t
#lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ±
lstm_cell_3/strided_slice_3StridedSlice$lstm_cell_3/ReadVariableOp_3:value:0*lstm_cell_3/strided_slice_3/stack:output:0,lstm_cell_3/strided_slice_3/stack_1:output:0,lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
lstm_cell_3/MatMul_7MatMullstm_cell_3/mul_7:z:0$lstm_cell_3/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/add_4AddV2lstm_cell_3/BiasAdd_3:output:0lstm_cell_3/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
lstm_cell_3/Tanh_1Tanhlstm_cell_3/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
lstm_cell_3/mul_10Mullstm_cell_3/Sigmoid_2:y:0lstm_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ó
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_3_split_readvariableop_resource+lstm_cell_3_split_1_readvariableop_resource#lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_24039*
condR
while_cond_24038*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^lstm_cell_3/ReadVariableOp^lstm_cell_3/ReadVariableOp_1^lstm_cell_3/ReadVariableOp_2^lstm_cell_3/ReadVariableOp_3!^lstm_cell_3/split/ReadVariableOp#^lstm_cell_3/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: : : 28
lstm_cell_3/ReadVariableOplstm_cell_3/ReadVariableOp2<
lstm_cell_3/ReadVariableOp_1lstm_cell_3/ReadVariableOp_12<
lstm_cell_3/ReadVariableOp_2lstm_cell_3/ReadVariableOp_22<
lstm_cell_3/ReadVariableOp_3lstm_cell_3/ReadVariableOp_32D
 lstm_cell_3/split/ReadVariableOp lstm_cell_3/split/ReadVariableOp2H
"lstm_cell_3/split_1/ReadVariableOp"lstm_cell_3/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
inputs/0
¶Ç
Ü
A__inference_lstm_2_layer_call_and_return_conditional_losses_25158

inputs<
)lstm_cell_3_split_readvariableop_resource:	d:
+lstm_cell_3_split_1_readvariableop_resource:	6
#lstm_cell_3_readvariableop_resource:	@
identity¢lstm_cell_3/ReadVariableOp¢lstm_cell_3/ReadVariableOp_1¢lstm_cell_3/ReadVariableOp_2¢lstm_cell_3/ReadVariableOp_3¢ lstm_cell_3/split/ReadVariableOp¢"lstm_cell_3/split_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:úÿÿÿÿÿÿÿÿÿdD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskc
lstm_cell_3/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:`
lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_3/ones_likeFill$lstm_cell_3/ones_like/Shape:output:0$lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd^
lstm_cell_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_3/dropout/MulMullstm_cell_3/ones_like:output:0"lstm_cell_3/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
lstm_cell_3/dropout/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:¤
0lstm_cell_3/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_3/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0g
"lstm_cell_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ê
 lstm_cell_3/dropout/GreaterEqualGreaterEqual9lstm_cell_3/dropout/random_uniform/RandomUniform:output:0+lstm_cell_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/dropout/CastCast$lstm_cell_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/dropout/Mul_1Mullstm_cell_3/dropout/Mul:z:0lstm_cell_3/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
lstm_cell_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_3/dropout_1/MulMullstm_cell_3/ones_like:output:0$lstm_cell_3/dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
lstm_cell_3/dropout_1/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:¨
2lstm_cell_3/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0i
$lstm_cell_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ð
"lstm_cell_3/dropout_1/GreaterEqualGreaterEqual;lstm_cell_3/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/dropout_1/CastCast&lstm_cell_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/dropout_1/Mul_1Mullstm_cell_3/dropout_1/Mul:z:0lstm_cell_3/dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
lstm_cell_3/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_3/dropout_2/MulMullstm_cell_3/ones_like:output:0$lstm_cell_3/dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
lstm_cell_3/dropout_2/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:¨
2lstm_cell_3/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0i
$lstm_cell_3/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ð
"lstm_cell_3/dropout_2/GreaterEqualGreaterEqual;lstm_cell_3/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/dropout_2/CastCast&lstm_cell_3/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/dropout_2/Mul_1Mullstm_cell_3/dropout_2/Mul:z:0lstm_cell_3/dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
lstm_cell_3/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_3/dropout_3/MulMullstm_cell_3/ones_like:output:0$lstm_cell_3/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
lstm_cell_3/dropout_3/ShapeShapelstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:¨
2lstm_cell_3/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0i
$lstm_cell_3/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ð
"lstm_cell_3/dropout_3/GreaterEqualGreaterEqual;lstm_cell_3/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/dropout_3/CastCast&lstm_cell_3/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/dropout_3/Mul_1Mullstm_cell_3/dropout_3/Mul:z:0lstm_cell_3/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
lstm_cell_3/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:b
lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
lstm_cell_3/ones_like_1Fill&lstm_cell_3/ones_like_1/Shape:output:0&lstm_cell_3/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
lstm_cell_3/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_3/dropout_4/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
lstm_cell_3/dropout_4/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:¨
2lstm_cell_3/dropout_4/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0i
$lstm_cell_3/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ð
"lstm_cell_3/dropout_4/GreaterEqualGreaterEqual;lstm_cell_3/dropout_4/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/dropout_4/CastCast&lstm_cell_3/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/dropout_4/Mul_1Mullstm_cell_3/dropout_4/Mul:z:0lstm_cell_3/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
lstm_cell_3/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_3/dropout_5/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
lstm_cell_3/dropout_5/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:¨
2lstm_cell_3/dropout_5/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0i
$lstm_cell_3/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ð
"lstm_cell_3/dropout_5/GreaterEqualGreaterEqual;lstm_cell_3/dropout_5/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/dropout_5/CastCast&lstm_cell_3/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/dropout_5/Mul_1Mullstm_cell_3/dropout_5/Mul:z:0lstm_cell_3/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
lstm_cell_3/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_3/dropout_6/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_6/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
lstm_cell_3/dropout_6/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:¨
2lstm_cell_3/dropout_6/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_6/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0i
$lstm_cell_3/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ð
"lstm_cell_3/dropout_6/GreaterEqualGreaterEqual;lstm_cell_3/dropout_6/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/dropout_6/CastCast&lstm_cell_3/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/dropout_6/Mul_1Mullstm_cell_3/dropout_6/Mul:z:0lstm_cell_3/dropout_6/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
lstm_cell_3/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_3/dropout_7/MulMul lstm_cell_3/ones_like_1:output:0$lstm_cell_3/dropout_7/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
lstm_cell_3/dropout_7/ShapeShape lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:¨
2lstm_cell_3/dropout_7/random_uniform/RandomUniformRandomUniform$lstm_cell_3/dropout_7/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0i
$lstm_cell_3/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ð
"lstm_cell_3/dropout_7/GreaterEqualGreaterEqual;lstm_cell_3/dropout_7/random_uniform/RandomUniform:output:0-lstm_cell_3/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/dropout_7/CastCast&lstm_cell_3/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/dropout_7/Mul_1Mullstm_cell_3/dropout_7/Mul:z:0lstm_cell_3/dropout_7/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/mulMulstrided_slice_2:output:0lstm_cell_3/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/mul_1Mulstrided_slice_2:output:0lstm_cell_3/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/mul_2Mulstrided_slice_2:output:0lstm_cell_3/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/mul_3Mulstrided_slice_2:output:0lstm_cell_3/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_cell_3/split/ReadVariableOpReadVariableOp)lstm_cell_3_split_readvariableop_resource*
_output_shapes
:	d*
dtype0Â
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0(lstm_cell_3/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:d@:d@:d@:d@*
	num_split
lstm_cell_3/MatMulMatMullstm_cell_3/mul:z:0lstm_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/MatMul_1MatMullstm_cell_3/mul_1:z:0lstm_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/MatMul_2MatMullstm_cell_3/mul_2:z:0lstm_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/MatMul_3MatMullstm_cell_3/mul_3:z:0lstm_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
"lstm_cell_3/split_1/ReadVariableOpReadVariableOp+lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0¸
lstm_cell_3/split_1Split&lstm_cell_3/split_1/split_dim:output:0*lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split
lstm_cell_3/BiasAddBiasAddlstm_cell_3/MatMul:product:0lstm_cell_3/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/BiasAdd_1BiasAddlstm_cell_3/MatMul_1:product:0lstm_cell_3/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/BiasAdd_2BiasAddlstm_cell_3/MatMul_2:product:0lstm_cell_3/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/BiasAdd_3BiasAddlstm_cell_3/MatMul_3:product:0lstm_cell_3/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
lstm_cell_3/mul_4Mulzeros:output:0lstm_cell_3/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
lstm_cell_3/mul_5Mulzeros:output:0lstm_cell_3/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
lstm_cell_3/mul_6Mulzeros:output:0lstm_cell_3/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
lstm_cell_3/mul_7Mulzeros:output:0lstm_cell_3/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/ReadVariableOpReadVariableOp#lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0p
lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   r
!lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
lstm_cell_3/strided_sliceStridedSlice"lstm_cell_3/ReadVariableOp:value:0(lstm_cell_3/strided_slice/stack:output:0*lstm_cell_3/strided_slice/stack_1:output:0*lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
lstm_cell_3/MatMul_4MatMullstm_cell_3/mul_4:z:0"lstm_cell_3/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/addAddV2lstm_cell_3/BiasAdd:output:0lstm_cell_3/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
lstm_cell_3/SigmoidSigmoidlstm_cell_3/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/ReadVariableOp_1ReadVariableOp#lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0r
!lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   t
#lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ±
lstm_cell_3/strided_slice_1StridedSlice$lstm_cell_3/ReadVariableOp_1:value:0*lstm_cell_3/strided_slice_1/stack:output:0,lstm_cell_3/strided_slice_1/stack_1:output:0,lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
lstm_cell_3/MatMul_5MatMullstm_cell_3/mul_5:z:0$lstm_cell_3/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/add_1AddV2lstm_cell_3/BiasAdd_1:output:0lstm_cell_3/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
lstm_cell_3/mul_8Mullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/ReadVariableOp_2ReadVariableOp#lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0r
!lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   t
#lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ±
lstm_cell_3/strided_slice_2StridedSlice$lstm_cell_3/ReadVariableOp_2:value:0*lstm_cell_3/strided_slice_2/stack:output:0,lstm_cell_3/strided_slice_2/stack_1:output:0,lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
lstm_cell_3/MatMul_6MatMullstm_cell_3/mul_6:z:0$lstm_cell_3/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/add_2AddV2lstm_cell_3/BiasAdd_2:output:0lstm_cell_3/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
lstm_cell_3/TanhTanhlstm_cell_3/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
lstm_cell_3/mul_9Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
lstm_cell_3/add_3AddV2lstm_cell_3/mul_8:z:0lstm_cell_3/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/ReadVariableOp_3ReadVariableOp#lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0r
!lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   t
#lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ±
lstm_cell_3/strided_slice_3StridedSlice$lstm_cell_3/ReadVariableOp_3:value:0*lstm_cell_3/strided_slice_3/stack:output:0,lstm_cell_3/strided_slice_3/stack_1:output:0,lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
lstm_cell_3/MatMul_7MatMullstm_cell_3/mul_7:z:0$lstm_cell_3/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/add_4AddV2lstm_cell_3/BiasAdd_3:output:0lstm_cell_3/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
lstm_cell_3/Tanh_1Tanhlstm_cell_3/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
lstm_cell_3/mul_10Mullstm_cell_3/Sigmoid_2:y:0lstm_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ó
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_3_split_readvariableop_resource+lstm_cell_3_split_1_readvariableop_resource#lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_24960*
condR
while_cond_24959*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ã
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:úÿÿÿÿÿÿÿÿÿ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿú@[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^lstm_cell_3/ReadVariableOp^lstm_cell_3/ReadVariableOp_1^lstm_cell_3/ReadVariableOp_2^lstm_cell_3/ReadVariableOp_3!^lstm_cell_3/split/ReadVariableOp#^lstm_cell_3/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿúd: : : 28
lstm_cell_3/ReadVariableOplstm_cell_3/ReadVariableOp2<
lstm_cell_3/ReadVariableOp_1lstm_cell_3/ReadVariableOp_12<
lstm_cell_3/ReadVariableOp_2lstm_cell_3/ReadVariableOp_22<
lstm_cell_3/ReadVariableOp_3lstm_cell_3/ReadVariableOp_32D
 lstm_cell_3/split/ReadVariableOp lstm_cell_3/split/ReadVariableOp2H
"lstm_cell_3/split_1/ReadVariableOp"lstm_cell_3/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúd
 
_user_specified_nameinputs
¦	
£
F__inference_embedding_4_layer_call_and_return_conditional_losses_23886

inputs)
embedding_lookup_23880:	Äd
identity¢embedding_lookupV
CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿúº
embedding_lookupResourceGatherembedding_lookup_23880Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/23880*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúd*
dtype0¢
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/23880*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúd
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúdx
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúdY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:ÿÿÿÿÿÿÿÿÿú: 2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs
°
¾
while_cond_22391
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_22391___redundant_placeholder03
/while_while_cond_22391___redundant_placeholder13
/while_while_cond_22391___redundant_placeholder23
/while_while_cond_22391___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
¾~
©
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_25460

inputs
states_0
states_10
split_readvariableop_resource:	d.
split_1_readvariableop_resource:	*
readvariableop_resource:	@
identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?p
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdO
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?t
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?t
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?t
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?v
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@S
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0]
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?v
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@S
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0]
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?v
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@S
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0]
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?v
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@S
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0]
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	d*
dtype0
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:d@:d@:d@:d@*
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@]
mul_4Mulstates_0dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@]
mul_5Mulstates_0dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@]
mul_6Mulstates_0dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@]
mul_7Mulstates_0dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ë
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskg
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      õ
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      õ
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@V
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      õ
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@K
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@À
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/1
¾

'__inference_dense_4_layer_call_fn_25167

inputs
unknown:@ 
	unknown_0: 
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_22545o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
õ
³
&__inference_lstm_2_layer_call_fn_23919

inputs
unknown:	d
	unknown_0:	
	unknown_1:	@
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_22526o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿúd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúd
 
_user_specified_nameinputs
D
©
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_25314

inputs
states_0
states_10
split_readvariableop_resource:	d.
split_1_readvariableop_resource:	*
readvariableop_resource:	@
identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
mul_1Mulinputsones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
mul_2Mulinputsones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
mul_3Mulinputsones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	d*
dtype0
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:d@:d@:d@:d@*
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@^
mul_4Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@^
mul_5Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@^
mul_6Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@^
mul_7Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ë
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskg
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      õ
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      õ
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@V
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      õ
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@K
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@À
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/1
Â
	
while_body_22794
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_3_split_readvariableop_resource_0:	dB
3while_lstm_cell_3_split_1_readvariableop_resource_0:	>
+while_lstm_cell_3_readvariableop_resource_0:	@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_3_split_readvariableop_resource:	d@
1while_lstm_cell_3_split_1_readvariableop_resource:	<
)while_lstm_cell_3_readvariableop_resource:	@¢ while/lstm_cell_3/ReadVariableOp¢"while/lstm_cell_3/ReadVariableOp_1¢"while/lstm_cell_3/ReadVariableOp_2¢"while/lstm_cell_3/ReadVariableOp_3¢&while/lstm_cell_3/split/ReadVariableOp¢(while/lstm_cell_3/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
!while/lstm_cell_3/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?­
while/lstm_cell_3/ones_likeFill*while/lstm_cell_3/ones_like/Shape:output:0*while/lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
while/lstm_cell_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¦
while/lstm_cell_3/dropout/MulMul$while/lstm_cell_3/ones_like:output:0(while/lstm_cell_3/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/lstm_cell_3/dropout/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:°
6while/lstm_cell_3/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_3/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0m
(while/lstm_cell_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ü
&while/lstm_cell_3/dropout/GreaterEqualGreaterEqual?while/lstm_cell_3/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_3/dropout/CastCast*while/lstm_cell_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_3/dropout/Mul_1Mul!while/lstm_cell_3/dropout/Mul:z:0"while/lstm_cell_3/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
!while/lstm_cell_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?ª
while/lstm_cell_3/dropout_1/MulMul$while/lstm_cell_3/ones_like:output:0*while/lstm_cell_3/dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
!while/lstm_cell_3/dropout_1/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:´
8while/lstm_cell_3/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0o
*while/lstm_cell_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=â
(while/lstm_cell_3/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 while/lstm_cell_3/dropout_1/CastCast,while/lstm_cell_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¥
!while/lstm_cell_3/dropout_1/Mul_1Mul#while/lstm_cell_3/dropout_1/Mul:z:0$while/lstm_cell_3/dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
!while/lstm_cell_3/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?ª
while/lstm_cell_3/dropout_2/MulMul$while/lstm_cell_3/ones_like:output:0*while/lstm_cell_3/dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
!while/lstm_cell_3/dropout_2/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:´
8while/lstm_cell_3/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0o
*while/lstm_cell_3/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=â
(while/lstm_cell_3/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 while/lstm_cell_3/dropout_2/CastCast,while/lstm_cell_3/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¥
!while/lstm_cell_3/dropout_2/Mul_1Mul#while/lstm_cell_3/dropout_2/Mul:z:0$while/lstm_cell_3/dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
!while/lstm_cell_3/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?ª
while/lstm_cell_3/dropout_3/MulMul$while/lstm_cell_3/ones_like:output:0*while/lstm_cell_3/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
!while/lstm_cell_3/dropout_3/ShapeShape$while/lstm_cell_3/ones_like:output:0*
T0*
_output_shapes
:´
8while/lstm_cell_3/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0o
*while/lstm_cell_3/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=â
(while/lstm_cell_3/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 while/lstm_cell_3/dropout_3/CastCast,while/lstm_cell_3/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¥
!while/lstm_cell_3/dropout_3/Mul_1Mul#while/lstm_cell_3/dropout_3/Mul:z:0$while/lstm_cell_3/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
#while/lstm_cell_3/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:h
#while/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
while/lstm_cell_3/ones_like_1Fill,while/lstm_cell_3/ones_like_1/Shape:output:0,while/lstm_cell_3/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@f
!while/lstm_cell_3/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¬
while/lstm_cell_3/dropout_4/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
!while/lstm_cell_3/dropout_4/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:´
8while/lstm_cell_3/dropout_4/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0o
*while/lstm_cell_3/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=â
(while/lstm_cell_3/dropout_4/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_4/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 while/lstm_cell_3/dropout_4/CastCast,while/lstm_cell_3/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
!while/lstm_cell_3/dropout_4/Mul_1Mul#while/lstm_cell_3/dropout_4/Mul:z:0$while/lstm_cell_3/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@f
!while/lstm_cell_3/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¬
while/lstm_cell_3/dropout_5/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
!while/lstm_cell_3/dropout_5/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:´
8while/lstm_cell_3/dropout_5/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0o
*while/lstm_cell_3/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=â
(while/lstm_cell_3/dropout_5/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_5/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 while/lstm_cell_3/dropout_5/CastCast,while/lstm_cell_3/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
!while/lstm_cell_3/dropout_5/Mul_1Mul#while/lstm_cell_3/dropout_5/Mul:z:0$while/lstm_cell_3/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@f
!while/lstm_cell_3/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¬
while/lstm_cell_3/dropout_6/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_6/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
!while/lstm_cell_3/dropout_6/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:´
8while/lstm_cell_3/dropout_6/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_6/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0o
*while/lstm_cell_3/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=â
(while/lstm_cell_3/dropout_6/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_6/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 while/lstm_cell_3/dropout_6/CastCast,while/lstm_cell_3/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
!while/lstm_cell_3/dropout_6/Mul_1Mul#while/lstm_cell_3/dropout_6/Mul:z:0$while/lstm_cell_3/dropout_6/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@f
!while/lstm_cell_3/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¬
while/lstm_cell_3/dropout_7/MulMul&while/lstm_cell_3/ones_like_1:output:0*while/lstm_cell_3/dropout_7/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
!while/lstm_cell_3/dropout_7/ShapeShape&while/lstm_cell_3/ones_like_1:output:0*
T0*
_output_shapes
:´
8while/lstm_cell_3/dropout_7/random_uniform/RandomUniformRandomUniform*while/lstm_cell_3/dropout_7/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0o
*while/lstm_cell_3/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=â
(while/lstm_cell_3/dropout_7/GreaterEqualGreaterEqualAwhile/lstm_cell_3/dropout_7/random_uniform/RandomUniform:output:03while/lstm_cell_3/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 while/lstm_cell_3/dropout_7/CastCast,while/lstm_cell_3/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
!while/lstm_cell_3/dropout_7/Mul_1Mul#while/lstm_cell_3/dropout_7/Mul:z:0$while/lstm_cell_3/dropout_7/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
while/lstm_cell_3/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell_3/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd©
while/lstm_cell_3/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_3/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd©
while/lstm_cell_3/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_3/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd©
while/lstm_cell_3/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_3/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
&while/lstm_cell_3/split/ReadVariableOpReadVariableOp1while_lstm_cell_3_split_readvariableop_resource_0*
_output_shapes
:	d*
dtype0Ô
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0.while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:d@:d@:d@:d@*
	num_split
while/lstm_cell_3/MatMulMatMulwhile/lstm_cell_3/mul:z:0 while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/MatMul_1MatMulwhile/lstm_cell_3/mul_1:z:0 while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/MatMul_2MatMulwhile/lstm_cell_3/mul_2:z:0 while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/MatMul_3MatMulwhile/lstm_cell_3/mul_3:z:0 while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
#while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
(while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Ê
while/lstm_cell_3/split_1Split,while/lstm_cell_3/split_1/split_dim:output:00while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split
while/lstm_cell_3/BiasAddBiasAdd"while/lstm_cell_3/MatMul:product:0"while/lstm_cell_3/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
while/lstm_cell_3/BiasAdd_1BiasAdd$while/lstm_cell_3/MatMul_1:product:0"while/lstm_cell_3/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
while/lstm_cell_3/BiasAdd_2BiasAdd$while/lstm_cell_3/MatMul_2:product:0"while/lstm_cell_3/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
while/lstm_cell_3/BiasAdd_3BiasAdd$while/lstm_cell_3/MatMul_3:product:0"while/lstm_cell_3/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_4Mulwhile_placeholder_2%while/lstm_cell_3/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_5Mulwhile_placeholder_2%while/lstm_cell_3/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_6Mulwhile_placeholder_2%while/lstm_cell_3/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_7Mulwhile_placeholder_2%while/lstm_cell_3/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 while/lstm_cell_3/ReadVariableOpReadVariableOp+while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0v
%while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   x
'while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Å
while/lstm_cell_3/strided_sliceStridedSlice(while/lstm_cell_3/ReadVariableOp:value:0.while/lstm_cell_3/strided_slice/stack:output:00while/lstm_cell_3/strided_slice/stack_1:output:00while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_4MatMulwhile/lstm_cell_3/mul_4:z:0(while/lstm_cell_3/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/addAddV2"while/lstm_cell_3/BiasAdd:output:0$while/lstm_cell_3/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
while/lstm_cell_3/SigmoidSigmoidwhile/lstm_cell_3/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"while/lstm_cell_3/ReadVariableOp_1ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0x
'while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   z
)while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ï
!while/lstm_cell_3/strided_slice_1StridedSlice*while/lstm_cell_3/ReadVariableOp_1:value:00while/lstm_cell_3/strided_slice_1/stack:output:02while/lstm_cell_3/strided_slice_1/stack_1:output:02while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_5MatMulwhile/lstm_cell_3/mul_5:z:0*while/lstm_cell_3/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/add_1AddV2$while/lstm_cell_3/BiasAdd_1:output:0$while/lstm_cell_3/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@u
while/lstm_cell_3/Sigmoid_1Sigmoidwhile/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_8Mulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"while/lstm_cell_3/ReadVariableOp_2ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0x
'while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   z
)while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ï
!while/lstm_cell_3/strided_slice_2StridedSlice*while/lstm_cell_3/ReadVariableOp_2:value:00while/lstm_cell_3/strided_slice_2/stack:output:02while/lstm_cell_3/strided_slice_2/stack_1:output:02while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_6MatMulwhile/lstm_cell_3/mul_6:z:0*while/lstm_cell_3/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/add_2AddV2$while/lstm_cell_3/BiasAdd_2:output:0$while/lstm_cell_3/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
while/lstm_cell_3/TanhTanhwhile/lstm_cell_3/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_9Mulwhile/lstm_cell_3/Sigmoid:y:0while/lstm_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/add_3AddV2while/lstm_cell_3/mul_8:z:0while/lstm_cell_3/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"while/lstm_cell_3/ReadVariableOp_3ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0x
'while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   z
)while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ï
!while/lstm_cell_3/strided_slice_3StridedSlice*while/lstm_cell_3/ReadVariableOp_3:value:00while/lstm_cell_3/strided_slice_3/stack:output:02while/lstm_cell_3/strided_slice_3/stack_1:output:02while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
while/lstm_cell_3/MatMul_7MatMulwhile/lstm_cell_3/mul_7:z:0*while/lstm_cell_3/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/add_4AddV2$while/lstm_cell_3/BiasAdd_3:output:0$while/lstm_cell_3/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@u
while/lstm_cell_3/Sigmoid_2Sigmoidwhile/lstm_cell_3/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
while/lstm_cell_3/Tanh_1Tanhwhile/lstm_cell_3/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
while/lstm_cell_3/mul_10Mulwhile/lstm_cell_3/Sigmoid_2:y:0while/lstm_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Å
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_10:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒy
while/Identity_4Identitywhile/lstm_cell_3/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
while/Identity_5Identitywhile/lstm_cell_3/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@²

while/NoOpNoOp!^while/lstm_cell_3/ReadVariableOp#^while/lstm_cell_3/ReadVariableOp_1#^while/lstm_cell_3/ReadVariableOp_2#^while/lstm_cell_3/ReadVariableOp_3'^while/lstm_cell_3/split/ReadVariableOp)^while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_3_readvariableop_resource+while_lstm_cell_3_readvariableop_resource_0"h
1while_lstm_cell_3_split_1_readvariableop_resource3while_lstm_cell_3_split_1_readvariableop_resource_0"d
/while_lstm_cell_3_split_readvariableop_resource1while_lstm_cell_3_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : 2D
 while/lstm_cell_3/ReadVariableOp while/lstm_cell_3/ReadVariableOp2H
"while/lstm_cell_3/ReadVariableOp_1"while/lstm_cell_3/ReadVariableOp_12H
"while/lstm_cell_3/ReadVariableOp_2"while/lstm_cell_3/ReadVariableOp_22H
"while/lstm_cell_3/ReadVariableOp_3"while/lstm_cell_3/ReadVariableOp_32P
&while/lstm_cell_3/split/ReadVariableOp&while/lstm_cell_3/split/ReadVariableOp2T
(while/lstm_cell_3/split_1/ReadVariableOp(while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
×

lstm_2_while_body_23307*
&lstm_2_while_lstm_2_while_loop_counter0
,lstm_2_while_lstm_2_while_maximum_iterations
lstm_2_while_placeholder
lstm_2_while_placeholder_1
lstm_2_while_placeholder_2
lstm_2_while_placeholder_3)
%lstm_2_while_lstm_2_strided_slice_1_0e
alstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0K
8lstm_2_while_lstm_cell_3_split_readvariableop_resource_0:	dI
:lstm_2_while_lstm_cell_3_split_1_readvariableop_resource_0:	E
2lstm_2_while_lstm_cell_3_readvariableop_resource_0:	@
lstm_2_while_identity
lstm_2_while_identity_1
lstm_2_while_identity_2
lstm_2_while_identity_3
lstm_2_while_identity_4
lstm_2_while_identity_5'
#lstm_2_while_lstm_2_strided_slice_1c
_lstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensorI
6lstm_2_while_lstm_cell_3_split_readvariableop_resource:	dG
8lstm_2_while_lstm_cell_3_split_1_readvariableop_resource:	C
0lstm_2_while_lstm_cell_3_readvariableop_resource:	@¢'lstm_2/while/lstm_cell_3/ReadVariableOp¢)lstm_2/while/lstm_cell_3/ReadVariableOp_1¢)lstm_2/while/lstm_cell_3/ReadVariableOp_2¢)lstm_2/while/lstm_cell_3/ReadVariableOp_3¢-lstm_2/while/lstm_cell_3/split/ReadVariableOp¢/lstm_2/while/lstm_cell_3/split_1/ReadVariableOp
>lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   É
0lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0lstm_2_while_placeholderGlstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
(lstm_2/while/lstm_cell_3/ones_like/ShapeShape7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:m
(lstm_2/while/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Â
"lstm_2/while/lstm_cell_3/ones_likeFill1lstm_2/while/lstm_cell_3/ones_like/Shape:output:01lstm_2/while/lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
*lstm_2/while/lstm_cell_3/ones_like_1/ShapeShapelstm_2_while_placeholder_2*
T0*
_output_shapes
:o
*lstm_2/while/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?È
$lstm_2/while/lstm_cell_3/ones_like_1Fill3lstm_2/while/lstm_cell_3/ones_like_1/Shape:output:03lstm_2/while/lstm_cell_3/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@»
lstm_2/while/lstm_cell_3/mulMul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_2/while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd½
lstm_2/while/lstm_cell_3/mul_1Mul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_2/while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd½
lstm_2/while/lstm_cell_3/mul_2Mul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_2/while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd½
lstm_2/while/lstm_cell_3/mul_3Mul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_2/while/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdj
(lstm_2/while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :§
-lstm_2/while/lstm_cell_3/split/ReadVariableOpReadVariableOp8lstm_2_while_lstm_cell_3_split_readvariableop_resource_0*
_output_shapes
:	d*
dtype0é
lstm_2/while/lstm_cell_3/splitSplit1lstm_2/while/lstm_cell_3/split/split_dim:output:05lstm_2/while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:d@:d@:d@:d@*
	num_split¦
lstm_2/while/lstm_cell_3/MatMulMatMul lstm_2/while/lstm_cell_3/mul:z:0'lstm_2/while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ª
!lstm_2/while/lstm_cell_3/MatMul_1MatMul"lstm_2/while/lstm_cell_3/mul_1:z:0'lstm_2/while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ª
!lstm_2/while/lstm_cell_3/MatMul_2MatMul"lstm_2/while/lstm_cell_3/mul_2:z:0'lstm_2/while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ª
!lstm_2/while/lstm_cell_3/MatMul_3MatMul"lstm_2/while/lstm_cell_3/mul_3:z:0'lstm_2/while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
*lstm_2/while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : §
/lstm_2/while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp:lstm_2_while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0ß
 lstm_2/while/lstm_cell_3/split_1Split3lstm_2/while/lstm_cell_3/split_1/split_dim:output:07lstm_2/while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split³
 lstm_2/while/lstm_cell_3/BiasAddBiasAdd)lstm_2/while/lstm_cell_3/MatMul:product:0)lstm_2/while/lstm_cell_3/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@·
"lstm_2/while/lstm_cell_3/BiasAdd_1BiasAdd+lstm_2/while/lstm_cell_3/MatMul_1:product:0)lstm_2/while/lstm_cell_3/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@·
"lstm_2/while/lstm_cell_3/BiasAdd_2BiasAdd+lstm_2/while/lstm_cell_3/MatMul_2:product:0)lstm_2/while/lstm_cell_3/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@·
"lstm_2/while/lstm_cell_3/BiasAdd_3BiasAdd+lstm_2/while/lstm_cell_3/MatMul_3:product:0)lstm_2/while/lstm_cell_3/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
lstm_2/while/lstm_cell_3/mul_4Mullstm_2_while_placeholder_2-lstm_2/while/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
lstm_2/while/lstm_cell_3/mul_5Mullstm_2_while_placeholder_2-lstm_2/while/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
lstm_2/while/lstm_cell_3/mul_6Mullstm_2_while_placeholder_2-lstm_2/while/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
lstm_2/while/lstm_cell_3/mul_7Mullstm_2_while_placeholder_2-lstm_2/while/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
'lstm_2/while/lstm_cell_3/ReadVariableOpReadVariableOp2lstm_2_while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0}
,lstm_2/while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.lstm_2/while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   
.lstm_2/while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      è
&lstm_2/while/lstm_cell_3/strided_sliceStridedSlice/lstm_2/while/lstm_cell_3/ReadVariableOp:value:05lstm_2/while/lstm_cell_3/strided_slice/stack:output:07lstm_2/while/lstm_cell_3/strided_slice/stack_1:output:07lstm_2/while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask²
!lstm_2/while/lstm_cell_3/MatMul_4MatMul"lstm_2/while/lstm_cell_3/mul_4:z:0/lstm_2/while/lstm_cell_3/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¯
lstm_2/while/lstm_cell_3/addAddV2)lstm_2/while/lstm_cell_3/BiasAdd:output:0+lstm_2/while/lstm_cell_3/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 lstm_2/while/lstm_cell_3/SigmoidSigmoid lstm_2/while/lstm_cell_3/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)lstm_2/while/lstm_cell_3/ReadVariableOp_1ReadVariableOp2lstm_2_while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0
.lstm_2/while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   
0lstm_2/while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
0lstm_2/while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ò
(lstm_2/while/lstm_cell_3/strided_slice_1StridedSlice1lstm_2/while/lstm_cell_3/ReadVariableOp_1:value:07lstm_2/while/lstm_cell_3/strided_slice_1/stack:output:09lstm_2/while/lstm_cell_3/strided_slice_1/stack_1:output:09lstm_2/while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask´
!lstm_2/while/lstm_cell_3/MatMul_5MatMul"lstm_2/while/lstm_cell_3/mul_5:z:01lstm_2/while/lstm_cell_3/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@³
lstm_2/while/lstm_cell_3/add_1AddV2+lstm_2/while/lstm_cell_3/BiasAdd_1:output:0+lstm_2/while/lstm_cell_3/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"lstm_2/while/lstm_cell_3/Sigmoid_1Sigmoid"lstm_2/while/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/while/lstm_cell_3/mul_8Mul&lstm_2/while/lstm_cell_3/Sigmoid_1:y:0lstm_2_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)lstm_2/while/lstm_cell_3/ReadVariableOp_2ReadVariableOp2lstm_2_while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0
.lstm_2/while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
0lstm_2/while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   
0lstm_2/while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ò
(lstm_2/while/lstm_cell_3/strided_slice_2StridedSlice1lstm_2/while/lstm_cell_3/ReadVariableOp_2:value:07lstm_2/while/lstm_cell_3/strided_slice_2/stack:output:09lstm_2/while/lstm_cell_3/strided_slice_2/stack_1:output:09lstm_2/while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask´
!lstm_2/while/lstm_cell_3/MatMul_6MatMul"lstm_2/while/lstm_cell_3/mul_6:z:01lstm_2/while/lstm_cell_3/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@³
lstm_2/while/lstm_cell_3/add_2AddV2+lstm_2/while/lstm_cell_3/BiasAdd_2:output:0+lstm_2/while/lstm_cell_3/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
lstm_2/while/lstm_cell_3/TanhTanh"lstm_2/while/lstm_cell_3/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
lstm_2/while/lstm_cell_3/mul_9Mul$lstm_2/while/lstm_cell_3/Sigmoid:y:0!lstm_2/while/lstm_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡
lstm_2/while/lstm_cell_3/add_3AddV2"lstm_2/while/lstm_cell_3/mul_8:z:0"lstm_2/while/lstm_cell_3/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)lstm_2/while/lstm_cell_3/ReadVariableOp_3ReadVariableOp2lstm_2_while_lstm_cell_3_readvariableop_resource_0*
_output_shapes
:	@*
dtype0
.lstm_2/while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   
0lstm_2/while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
0lstm_2/while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ò
(lstm_2/while/lstm_cell_3/strided_slice_3StridedSlice1lstm_2/while/lstm_cell_3/ReadVariableOp_3:value:07lstm_2/while/lstm_cell_3/strided_slice_3/stack:output:09lstm_2/while/lstm_cell_3/strided_slice_3/stack_1:output:09lstm_2/while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask´
!lstm_2/while/lstm_cell_3/MatMul_7MatMul"lstm_2/while/lstm_cell_3/mul_7:z:01lstm_2/while/lstm_cell_3/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@³
lstm_2/while/lstm_cell_3/add_4AddV2+lstm_2/while/lstm_cell_3/BiasAdd_3:output:0+lstm_2/while/lstm_cell_3/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"lstm_2/while/lstm_cell_3/Sigmoid_2Sigmoid"lstm_2/while/lstm_cell_3/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@}
lstm_2/while/lstm_cell_3/Tanh_1Tanh"lstm_2/while/lstm_cell_3/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¥
lstm_2/while/lstm_cell_3/mul_10Mul&lstm_2/while/lstm_cell_3/Sigmoid_2:y:0#lstm_2/while/lstm_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@á
1lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_2_while_placeholder_1lstm_2_while_placeholder#lstm_2/while/lstm_cell_3/mul_10:z:0*
_output_shapes
: *
element_dtype0:éèÒT
lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_2/while/addAddV2lstm_2_while_placeholderlstm_2/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_2/while/add_1AddV2&lstm_2_while_lstm_2_while_loop_counterlstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_2/while/IdentityIdentitylstm_2/while/add_1:z:0^lstm_2/while/NoOp*
T0*
_output_shapes
: 
lstm_2/while/Identity_1Identity,lstm_2_while_lstm_2_while_maximum_iterations^lstm_2/while/NoOp*
T0*
_output_shapes
: n
lstm_2/while/Identity_2Identitylstm_2/while/add:z:0^lstm_2/while/NoOp*
T0*
_output_shapes
: ®
lstm_2/while/Identity_3IdentityAlstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_2/while/NoOp*
T0*
_output_shapes
: :éèÒ
lstm_2/while/Identity_4Identity#lstm_2/while/lstm_cell_3/mul_10:z:0^lstm_2/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_2/while/Identity_5Identity"lstm_2/while/lstm_cell_3/add_3:z:0^lstm_2/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ã
lstm_2/while/NoOpNoOp(^lstm_2/while/lstm_cell_3/ReadVariableOp*^lstm_2/while/lstm_cell_3/ReadVariableOp_1*^lstm_2/while/lstm_cell_3/ReadVariableOp_2*^lstm_2/while/lstm_cell_3/ReadVariableOp_3.^lstm_2/while/lstm_cell_3/split/ReadVariableOp0^lstm_2/while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_2_while_identitylstm_2/while/Identity:output:0";
lstm_2_while_identity_1 lstm_2/while/Identity_1:output:0";
lstm_2_while_identity_2 lstm_2/while/Identity_2:output:0";
lstm_2_while_identity_3 lstm_2/while/Identity_3:output:0";
lstm_2_while_identity_4 lstm_2/while/Identity_4:output:0";
lstm_2_while_identity_5 lstm_2/while/Identity_5:output:0"L
#lstm_2_while_lstm_2_strided_slice_1%lstm_2_while_lstm_2_strided_slice_1_0"f
0lstm_2_while_lstm_cell_3_readvariableop_resource2lstm_2_while_lstm_cell_3_readvariableop_resource_0"v
8lstm_2_while_lstm_cell_3_split_1_readvariableop_resource:lstm_2_while_lstm_cell_3_split_1_readvariableop_resource_0"r
6lstm_2_while_lstm_cell_3_split_readvariableop_resource8lstm_2_while_lstm_cell_3_split_readvariableop_resource_0"Ä
_lstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensoralstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : 2R
'lstm_2/while/lstm_cell_3/ReadVariableOp'lstm_2/while/lstm_cell_3/ReadVariableOp2V
)lstm_2/while/lstm_cell_3/ReadVariableOp_1)lstm_2/while/lstm_cell_3/ReadVariableOp_12V
)lstm_2/while/lstm_cell_3/ReadVariableOp_2)lstm_2/while/lstm_cell_3/ReadVariableOp_22V
)lstm_2/while/lstm_cell_3/ReadVariableOp_3)lstm_2/while/lstm_cell_3/ReadVariableOp_32^
-lstm_2/while/lstm_cell_3/split/ReadVariableOp-lstm_2/while/lstm_cell_3/split/ReadVariableOp2b
/lstm_2/while/lstm_cell_3/split_1/ReadVariableOp/lstm_2/while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
: 
ñ7
ü
A__inference_lstm_2_layer_call_and_return_conditional_losses_22254

inputs$
lstm_cell_3_22172:	d 
lstm_cell_3_22174:	$
lstm_cell_3_22176:	@
identity¢#lstm_cell_3/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskì
#lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_3_22172lstm_cell_3_22174lstm_cell_3_22176*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_22126n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¯
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_3_22172lstm_cell_3_22174lstm_cell_3_22176*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_22185*
condR
while_cond_22184*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
NoOpNoOp$^lstm_cell_3/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: : : 2J
#lstm_cell_3/StatefulPartitionedCall#lstm_cell_3/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
«Ê
Ê	
 __inference__wrapped_model_21749
embedding_4_inputB
/sequential_4_embedding_4_embedding_lookup_21490:	ÄdP
=sequential_4_lstm_2_lstm_cell_3_split_readvariableop_resource:	dN
?sequential_4_lstm_2_lstm_cell_3_split_1_readvariableop_resource:	J
7sequential_4_lstm_2_lstm_cell_3_readvariableop_resource:	@E
3sequential_4_dense_4_matmul_readvariableop_resource:@ B
4sequential_4_dense_4_biasadd_readvariableop_resource: E
3sequential_4_dense_5_matmul_readvariableop_resource: B
4sequential_4_dense_5_biasadd_readvariableop_resource:
identity¢+sequential_4/dense_4/BiasAdd/ReadVariableOp¢*sequential_4/dense_4/MatMul/ReadVariableOp¢+sequential_4/dense_5/BiasAdd/ReadVariableOp¢*sequential_4/dense_5/MatMul/ReadVariableOp¢)sequential_4/embedding_4/embedding_lookup¢.sequential_4/lstm_2/lstm_cell_3/ReadVariableOp¢0sequential_4/lstm_2/lstm_cell_3/ReadVariableOp_1¢0sequential_4/lstm_2/lstm_cell_3/ReadVariableOp_2¢0sequential_4/lstm_2/lstm_cell_3/ReadVariableOp_3¢4sequential_4/lstm_2/lstm_cell_3/split/ReadVariableOp¢6sequential_4/lstm_2/lstm_cell_3/split_1/ReadVariableOp¢sequential_4/lstm_2/whilez
sequential_4/embedding_4/CastCastembedding_4_input*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
)sequential_4/embedding_4/embedding_lookupResourceGather/sequential_4_embedding_4_embedding_lookup_21490!sequential_4/embedding_4/Cast:y:0*
Tindices0*B
_class8
64loc:@sequential_4/embedding_4/embedding_lookup/21490*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúd*
dtype0í
2sequential_4/embedding_4/embedding_lookup/IdentityIdentity2sequential_4/embedding_4/embedding_lookup:output:0*
T0*B
_class8
64loc:@sequential_4/embedding_4/embedding_lookup/21490*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúd´
4sequential_4/embedding_4/embedding_lookup/Identity_1Identity;sequential_4/embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúd
sequential_4/lstm_2/ShapeShape=sequential_4/embedding_4/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:q
'sequential_4/lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_4/lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_4/lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
!sequential_4/lstm_2/strided_sliceStridedSlice"sequential_4/lstm_2/Shape:output:00sequential_4/lstm_2/strided_slice/stack:output:02sequential_4/lstm_2/strided_slice/stack_1:output:02sequential_4/lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sequential_4/lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@¯
 sequential_4/lstm_2/zeros/packedPack*sequential_4/lstm_2/strided_slice:output:0+sequential_4/lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_4/lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
sequential_4/lstm_2/zerosFill)sequential_4/lstm_2/zeros/packed:output:0(sequential_4/lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@f
$sequential_4/lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@³
"sequential_4/lstm_2/zeros_1/packedPack*sequential_4/lstm_2/strided_slice:output:0-sequential_4/lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_4/lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
sequential_4/lstm_2/zeros_1Fill+sequential_4/lstm_2/zeros_1/packed:output:0*sequential_4/lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
"sequential_4/lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Í
sequential_4/lstm_2/transpose	Transpose=sequential_4/embedding_4/embedding_lookup/Identity_1:output:0+sequential_4/lstm_2/transpose/perm:output:0*
T0*,
_output_shapes
:úÿÿÿÿÿÿÿÿÿdl
sequential_4/lstm_2/Shape_1Shape!sequential_4/lstm_2/transpose:y:0*
T0*
_output_shapes
:s
)sequential_4/lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_4/lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_4/lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#sequential_4/lstm_2/strided_slice_1StridedSlice$sequential_4/lstm_2/Shape_1:output:02sequential_4/lstm_2/strided_slice_1/stack:output:04sequential_4/lstm_2/strided_slice_1/stack_1:output:04sequential_4/lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/sequential_4/lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿð
!sequential_4/lstm_2/TensorArrayV2TensorListReserve8sequential_4/lstm_2/TensorArrayV2/element_shape:output:0,sequential_4/lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Isequential_4/lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   
;sequential_4/lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_4/lstm_2/transpose:y:0Rsequential_4/lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒs
)sequential_4/lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_4/lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_4/lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Í
#sequential_4/lstm_2/strided_slice_2StridedSlice!sequential_4/lstm_2/transpose:y:02sequential_4/lstm_2/strided_slice_2/stack:output:04sequential_4/lstm_2/strided_slice_2/stack_1:output:04sequential_4/lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask
/sequential_4/lstm_2/lstm_cell_3/ones_like/ShapeShape,sequential_4/lstm_2/strided_slice_2:output:0*
T0*
_output_shapes
:t
/sequential_4/lstm_2/lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?×
)sequential_4/lstm_2/lstm_cell_3/ones_likeFill8sequential_4/lstm_2/lstm_cell_3/ones_like/Shape:output:08sequential_4/lstm_2/lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
1sequential_4/lstm_2/lstm_cell_3/ones_like_1/ShapeShape"sequential_4/lstm_2/zeros:output:0*
T0*
_output_shapes
:v
1sequential_4/lstm_2/lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ý
+sequential_4/lstm_2/lstm_cell_3/ones_like_1Fill:sequential_4/lstm_2/lstm_cell_3/ones_like_1/Shape:output:0:sequential_4/lstm_2/lstm_cell_3/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¾
#sequential_4/lstm_2/lstm_cell_3/mulMul,sequential_4/lstm_2/strided_slice_2:output:02sequential_4/lstm_2/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÀ
%sequential_4/lstm_2/lstm_cell_3/mul_1Mul,sequential_4/lstm_2/strided_slice_2:output:02sequential_4/lstm_2/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÀ
%sequential_4/lstm_2/lstm_cell_3/mul_2Mul,sequential_4/lstm_2/strided_slice_2:output:02sequential_4/lstm_2/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÀ
%sequential_4/lstm_2/lstm_cell_3/mul_3Mul,sequential_4/lstm_2/strided_slice_2:output:02sequential_4/lstm_2/lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
/sequential_4/lstm_2/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :³
4sequential_4/lstm_2/lstm_cell_3/split/ReadVariableOpReadVariableOp=sequential_4_lstm_2_lstm_cell_3_split_readvariableop_resource*
_output_shapes
:	d*
dtype0þ
%sequential_4/lstm_2/lstm_cell_3/splitSplit8sequential_4/lstm_2/lstm_cell_3/split/split_dim:output:0<sequential_4/lstm_2/lstm_cell_3/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:d@:d@:d@:d@*
	num_split»
&sequential_4/lstm_2/lstm_cell_3/MatMulMatMul'sequential_4/lstm_2/lstm_cell_3/mul:z:0.sequential_4/lstm_2/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¿
(sequential_4/lstm_2/lstm_cell_3/MatMul_1MatMul)sequential_4/lstm_2/lstm_cell_3/mul_1:z:0.sequential_4/lstm_2/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¿
(sequential_4/lstm_2/lstm_cell_3/MatMul_2MatMul)sequential_4/lstm_2/lstm_cell_3/mul_2:z:0.sequential_4/lstm_2/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¿
(sequential_4/lstm_2/lstm_cell_3/MatMul_3MatMul)sequential_4/lstm_2/lstm_cell_3/mul_3:z:0.sequential_4/lstm_2/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
1sequential_4/lstm_2/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ³
6sequential_4/lstm_2/lstm_cell_3/split_1/ReadVariableOpReadVariableOp?sequential_4_lstm_2_lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ô
'sequential_4/lstm_2/lstm_cell_3/split_1Split:sequential_4/lstm_2/lstm_cell_3/split_1/split_dim:output:0>sequential_4/lstm_2/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitÈ
'sequential_4/lstm_2/lstm_cell_3/BiasAddBiasAdd0sequential_4/lstm_2/lstm_cell_3/MatMul:product:00sequential_4/lstm_2/lstm_cell_3/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ì
)sequential_4/lstm_2/lstm_cell_3/BiasAdd_1BiasAdd2sequential_4/lstm_2/lstm_cell_3/MatMul_1:product:00sequential_4/lstm_2/lstm_cell_3/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ì
)sequential_4/lstm_2/lstm_cell_3/BiasAdd_2BiasAdd2sequential_4/lstm_2/lstm_cell_3/MatMul_2:product:00sequential_4/lstm_2/lstm_cell_3/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ì
)sequential_4/lstm_2/lstm_cell_3/BiasAdd_3BiasAdd2sequential_4/lstm_2/lstm_cell_3/MatMul_3:product:00sequential_4/lstm_2/lstm_cell_3/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¸
%sequential_4/lstm_2/lstm_cell_3/mul_4Mul"sequential_4/lstm_2/zeros:output:04sequential_4/lstm_2/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¸
%sequential_4/lstm_2/lstm_cell_3/mul_5Mul"sequential_4/lstm_2/zeros:output:04sequential_4/lstm_2/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¸
%sequential_4/lstm_2/lstm_cell_3/mul_6Mul"sequential_4/lstm_2/zeros:output:04sequential_4/lstm_2/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¸
%sequential_4/lstm_2/lstm_cell_3/mul_7Mul"sequential_4/lstm_2/zeros:output:04sequential_4/lstm_2/lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@§
.sequential_4/lstm_2/lstm_cell_3/ReadVariableOpReadVariableOp7sequential_4_lstm_2_lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0
3sequential_4/lstm_2/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
5sequential_4/lstm_2/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   
5sequential_4/lstm_2/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
-sequential_4/lstm_2/lstm_cell_3/strided_sliceStridedSlice6sequential_4/lstm_2/lstm_cell_3/ReadVariableOp:value:0<sequential_4/lstm_2/lstm_cell_3/strided_slice/stack:output:0>sequential_4/lstm_2/lstm_cell_3/strided_slice/stack_1:output:0>sequential_4/lstm_2/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskÇ
(sequential_4/lstm_2/lstm_cell_3/MatMul_4MatMul)sequential_4/lstm_2/lstm_cell_3/mul_4:z:06sequential_4/lstm_2/lstm_cell_3/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
#sequential_4/lstm_2/lstm_cell_3/addAddV20sequential_4/lstm_2/lstm_cell_3/BiasAdd:output:02sequential_4/lstm_2/lstm_cell_3/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
'sequential_4/lstm_2/lstm_cell_3/SigmoidSigmoid'sequential_4/lstm_2/lstm_cell_3/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@©
0sequential_4/lstm_2/lstm_cell_3/ReadVariableOp_1ReadVariableOp7sequential_4_lstm_2_lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0
5sequential_4/lstm_2/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   
7sequential_4/lstm_2/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
7sequential_4/lstm_2/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
/sequential_4/lstm_2/lstm_cell_3/strided_slice_1StridedSlice8sequential_4/lstm_2/lstm_cell_3/ReadVariableOp_1:value:0>sequential_4/lstm_2/lstm_cell_3/strided_slice_1/stack:output:0@sequential_4/lstm_2/lstm_cell_3/strided_slice_1/stack_1:output:0@sequential_4/lstm_2/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskÉ
(sequential_4/lstm_2/lstm_cell_3/MatMul_5MatMul)sequential_4/lstm_2/lstm_cell_3/mul_5:z:08sequential_4/lstm_2/lstm_cell_3/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@È
%sequential_4/lstm_2/lstm_cell_3/add_1AddV22sequential_4/lstm_2/lstm_cell_3/BiasAdd_1:output:02sequential_4/lstm_2/lstm_cell_3/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)sequential_4/lstm_2/lstm_cell_3/Sigmoid_1Sigmoid)sequential_4/lstm_2/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@³
%sequential_4/lstm_2/lstm_cell_3/mul_8Mul-sequential_4/lstm_2/lstm_cell_3/Sigmoid_1:y:0$sequential_4/lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@©
0sequential_4/lstm_2/lstm_cell_3/ReadVariableOp_2ReadVariableOp7sequential_4_lstm_2_lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0
5sequential_4/lstm_2/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
7sequential_4/lstm_2/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   
7sequential_4/lstm_2/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
/sequential_4/lstm_2/lstm_cell_3/strided_slice_2StridedSlice8sequential_4/lstm_2/lstm_cell_3/ReadVariableOp_2:value:0>sequential_4/lstm_2/lstm_cell_3/strided_slice_2/stack:output:0@sequential_4/lstm_2/lstm_cell_3/strided_slice_2/stack_1:output:0@sequential_4/lstm_2/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskÉ
(sequential_4/lstm_2/lstm_cell_3/MatMul_6MatMul)sequential_4/lstm_2/lstm_cell_3/mul_6:z:08sequential_4/lstm_2/lstm_cell_3/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@È
%sequential_4/lstm_2/lstm_cell_3/add_2AddV22sequential_4/lstm_2/lstm_cell_3/BiasAdd_2:output:02sequential_4/lstm_2/lstm_cell_3/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$sequential_4/lstm_2/lstm_cell_3/TanhTanh)sequential_4/lstm_2/lstm_cell_3/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@µ
%sequential_4/lstm_2/lstm_cell_3/mul_9Mul+sequential_4/lstm_2/lstm_cell_3/Sigmoid:y:0(sequential_4/lstm_2/lstm_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¶
%sequential_4/lstm_2/lstm_cell_3/add_3AddV2)sequential_4/lstm_2/lstm_cell_3/mul_8:z:0)sequential_4/lstm_2/lstm_cell_3/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@©
0sequential_4/lstm_2/lstm_cell_3/ReadVariableOp_3ReadVariableOp7sequential_4_lstm_2_lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0
5sequential_4/lstm_2/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   
7sequential_4/lstm_2/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
7sequential_4/lstm_2/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
/sequential_4/lstm_2/lstm_cell_3/strided_slice_3StridedSlice8sequential_4/lstm_2/lstm_cell_3/ReadVariableOp_3:value:0>sequential_4/lstm_2/lstm_cell_3/strided_slice_3/stack:output:0@sequential_4/lstm_2/lstm_cell_3/strided_slice_3/stack_1:output:0@sequential_4/lstm_2/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskÉ
(sequential_4/lstm_2/lstm_cell_3/MatMul_7MatMul)sequential_4/lstm_2/lstm_cell_3/mul_7:z:08sequential_4/lstm_2/lstm_cell_3/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@È
%sequential_4/lstm_2/lstm_cell_3/add_4AddV22sequential_4/lstm_2/lstm_cell_3/BiasAdd_3:output:02sequential_4/lstm_2/lstm_cell_3/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)sequential_4/lstm_2/lstm_cell_3/Sigmoid_2Sigmoid)sequential_4/lstm_2/lstm_cell_3/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
&sequential_4/lstm_2/lstm_cell_3/Tanh_1Tanh)sequential_4/lstm_2/lstm_cell_3/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
&sequential_4/lstm_2/lstm_cell_3/mul_10Mul-sequential_4/lstm_2/lstm_cell_3/Sigmoid_2:y:0*sequential_4/lstm_2/lstm_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
1sequential_4/lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ô
#sequential_4/lstm_2/TensorArrayV2_1TensorListReserve:sequential_4/lstm_2/TensorArrayV2_1/element_shape:output:0,sequential_4/lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒZ
sequential_4/lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,sequential_4/lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿh
&sequential_4/lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
sequential_4/lstm_2/whileWhile/sequential_4/lstm_2/while/loop_counter:output:05sequential_4/lstm_2/while/maximum_iterations:output:0!sequential_4/lstm_2/time:output:0,sequential_4/lstm_2/TensorArrayV2_1:handle:0"sequential_4/lstm_2/zeros:output:0$sequential_4/lstm_2/zeros_1:output:0,sequential_4/lstm_2/strided_slice_1:output:0Ksequential_4/lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0=sequential_4_lstm_2_lstm_cell_3_split_readvariableop_resource?sequential_4_lstm_2_lstm_cell_3_split_1_readvariableop_resource7sequential_4_lstm_2_lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *0
body(R&
$sequential_4_lstm_2_while_body_21601*0
cond(R&
$sequential_4_lstm_2_while_cond_21600*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
Dsequential_4/lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ÿ
6sequential_4/lstm_2/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_4/lstm_2/while:output:3Msequential_4/lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:úÿÿÿÿÿÿÿÿÿ@*
element_dtype0|
)sequential_4/lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿu
+sequential_4/lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+sequential_4/lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ë
#sequential_4/lstm_2/strided_slice_3StridedSlice?sequential_4/lstm_2/TensorArrayV2Stack/TensorListStack:tensor:02sequential_4/lstm_2/strided_slice_3/stack:output:04sequential_4/lstm_2/strided_slice_3/stack_1:output:04sequential_4/lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_masky
$sequential_4/lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ó
sequential_4/lstm_2/transpose_1	Transpose?sequential_4/lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_4/lstm_2/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿú@o
sequential_4/lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
*sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_4_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0¹
sequential_4/dense_4/MatMulMatMul,sequential_4/lstm_2/strided_slice_3:output:02sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
+sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0µ
sequential_4/dense_4/BiasAddBiasAdd%sequential_4/dense_4/MatMul:product:03sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
sequential_4/dense_4/ReluRelu%sequential_4/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
*sequential_4/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype0´
sequential_4/dense_5/MatMulMatMul'sequential_4/dense_4/Relu:activations:02sequential_4/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_4/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
sequential_4/dense_5/BiasAddBiasAdd%sequential_4/dense_5/MatMul:product:03sequential_4/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_4/dense_5/SigmoidSigmoid%sequential_4/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
IdentityIdentity sequential_4/dense_5/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ
NoOpNoOp,^sequential_4/dense_4/BiasAdd/ReadVariableOp+^sequential_4/dense_4/MatMul/ReadVariableOp,^sequential_4/dense_5/BiasAdd/ReadVariableOp+^sequential_4/dense_5/MatMul/ReadVariableOp*^sequential_4/embedding_4/embedding_lookup/^sequential_4/lstm_2/lstm_cell_3/ReadVariableOp1^sequential_4/lstm_2/lstm_cell_3/ReadVariableOp_11^sequential_4/lstm_2/lstm_cell_3/ReadVariableOp_21^sequential_4/lstm_2/lstm_cell_3/ReadVariableOp_35^sequential_4/lstm_2/lstm_cell_3/split/ReadVariableOp7^sequential_4/lstm_2/lstm_cell_3/split_1/ReadVariableOp^sequential_4/lstm_2/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿú: : : : : : : : 2Z
+sequential_4/dense_4/BiasAdd/ReadVariableOp+sequential_4/dense_4/BiasAdd/ReadVariableOp2X
*sequential_4/dense_4/MatMul/ReadVariableOp*sequential_4/dense_4/MatMul/ReadVariableOp2Z
+sequential_4/dense_5/BiasAdd/ReadVariableOp+sequential_4/dense_5/BiasAdd/ReadVariableOp2X
*sequential_4/dense_5/MatMul/ReadVariableOp*sequential_4/dense_5/MatMul/ReadVariableOp2V
)sequential_4/embedding_4/embedding_lookup)sequential_4/embedding_4/embedding_lookup2`
.sequential_4/lstm_2/lstm_cell_3/ReadVariableOp.sequential_4/lstm_2/lstm_cell_3/ReadVariableOp2d
0sequential_4/lstm_2/lstm_cell_3/ReadVariableOp_10sequential_4/lstm_2/lstm_cell_3/ReadVariableOp_12d
0sequential_4/lstm_2/lstm_cell_3/ReadVariableOp_20sequential_4/lstm_2/lstm_cell_3/ReadVariableOp_22d
0sequential_4/lstm_2/lstm_cell_3/ReadVariableOp_30sequential_4/lstm_2/lstm_cell_3/ReadVariableOp_32l
4sequential_4/lstm_2/lstm_cell_3/split/ReadVariableOp4sequential_4/lstm_2/lstm_cell_3/split/ReadVariableOp2p
6sequential_4/lstm_2/lstm_cell_3/split_1/ReadVariableOp6sequential_4/lstm_2/lstm_cell_3/split_1/ReadVariableOp26
sequential_4/lstm_2/whilesequential_4/lstm_2/while:[ W
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
+
_user_specified_nameembedding_4_input
¦	
£
F__inference_embedding_4_layer_call_and_return_conditional_losses_22280

inputs)
embedding_lookup_22274:	Äd
identity¢embedding_lookupV
CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿúº
embedding_lookupResourceGatherembedding_lookup_22274Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/22274*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúd*
dtype0¢
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/22274*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúd
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúdx
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúdY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:ÿÿÿÿÿÿÿÿÿú: 2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
 
_user_specified_nameinputs
ô	
Ê
lstm_2_while_cond_23633*
&lstm_2_while_lstm_2_while_loop_counter0
,lstm_2_while_lstm_2_while_maximum_iterations
lstm_2_while_placeholder
lstm_2_while_placeholder_1
lstm_2_while_placeholder_2
lstm_2_while_placeholder_3,
(lstm_2_while_less_lstm_2_strided_slice_1A
=lstm_2_while_lstm_2_while_cond_23633___redundant_placeholder0A
=lstm_2_while_lstm_2_while_cond_23633___redundant_placeholder1A
=lstm_2_while_lstm_2_while_cond_23633___redundant_placeholder2A
=lstm_2_while_lstm_2_while_cond_23633___redundant_placeholder3
lstm_2_while_identity
~
lstm_2/while/LessLesslstm_2_while_placeholder(lstm_2_while_less_lstm_2_strided_slice_1*
T0*
_output_shapes
: Y
lstm_2/while/IdentityIdentitylstm_2/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_2_while_identitylstm_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
°
¾
while_cond_24959
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_24959___redundant_placeholder03
/while_while_cond_24959___redundant_placeholder13
/while_while_cond_24959___redundant_placeholder23
/while_while_cond_24959___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@:

_output_shapes
: :

_output_shapes
:
ò
ý
G__inference_sequential_4_layer_call_and_return_conditional_losses_23144
embedding_4_input$
embedding_4_23123:	Äd
lstm_2_23126:	d
lstm_2_23128:	
lstm_2_23130:	@
dense_4_23133:@ 
dense_4_23135: 
dense_5_23138: 
dense_5_23140:
identity¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢#embedding_4/StatefulPartitionedCall¢lstm_2/StatefulPartitionedCallô
#embedding_4/StatefulPartitionedCallStatefulPartitionedCallembedding_4_inputembedding_4_23123*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúd*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_embedding_4_layer_call_and_return_conditional_losses_22280
lstm_2/StatefulPartitionedCallStatefulPartitionedCall,embedding_4/StatefulPartitionedCall:output:0lstm_2_23126lstm_2_23128lstm_2_23130*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_22992
dense_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0dense_4_23133dense_4_23135*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_22545
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_23138dense_5_23140*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_22562w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
NoOpNoOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall$^embedding_4/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿú: : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2J
#embedding_4/StatefulPartitionedCall#embedding_4/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall:[ W
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
+
_user_specified_nameembedding_4_input
D
§
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_21866

inputs

states
states_10
split_readvariableop_resource:	d.
split_1_readvariableop_resource:	*
readvariableop_resource:	@
identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdG
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
mul_1Mulinputsones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
mul_2Mulinputsones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
mul_3Mulinputsones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	d*
dtype0
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:d@:d@:d@:d@*
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\
mul_4Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\
mul_5Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\
mul_6Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\
mul_7Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ë
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskg
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      õ
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      õ
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@V
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	@*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      õ
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@K
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@À
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_namestates

Ü
A__inference_lstm_2_layer_call_and_return_conditional_losses_24787

inputs<
)lstm_cell_3_split_readvariableop_resource:	d:
+lstm_cell_3_split_1_readvariableop_resource:	6
#lstm_cell_3_readvariableop_resource:	@
identity¢lstm_cell_3/ReadVariableOp¢lstm_cell_3/ReadVariableOp_1¢lstm_cell_3/ReadVariableOp_2¢lstm_cell_3/ReadVariableOp_3¢ lstm_cell_3/split/ReadVariableOp¢"lstm_cell_3/split_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:úÿÿÿÿÿÿÿÿÿdD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskc
lstm_cell_3/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:`
lstm_cell_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_3/ones_likeFill$lstm_cell_3/ones_like/Shape:output:0$lstm_cell_3/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
lstm_cell_3/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:b
lstm_cell_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
lstm_cell_3/ones_like_1Fill&lstm_cell_3/ones_like_1/Shape:output:0&lstm_cell_3/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/mulMulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/mul_1Mulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/mul_2Mulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_3/mul_3Mulstrided_slice_2:output:0lstm_cell_3/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_cell_3/split/ReadVariableOpReadVariableOp)lstm_cell_3_split_readvariableop_resource*
_output_shapes
:	d*
dtype0Â
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0(lstm_cell_3/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:d@:d@:d@:d@*
	num_split
lstm_cell_3/MatMulMatMullstm_cell_3/mul:z:0lstm_cell_3/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/MatMul_1MatMullstm_cell_3/mul_1:z:0lstm_cell_3/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/MatMul_2MatMullstm_cell_3/mul_2:z:0lstm_cell_3/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/MatMul_3MatMullstm_cell_3/mul_3:z:0lstm_cell_3/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
"lstm_cell_3/split_1/ReadVariableOpReadVariableOp+lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0¸
lstm_cell_3/split_1Split&lstm_cell_3/split_1/split_dim:output:0*lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split
lstm_cell_3/BiasAddBiasAddlstm_cell_3/MatMul:product:0lstm_cell_3/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/BiasAdd_1BiasAddlstm_cell_3/MatMul_1:product:0lstm_cell_3/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/BiasAdd_2BiasAddlstm_cell_3/MatMul_2:product:0lstm_cell_3/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/BiasAdd_3BiasAddlstm_cell_3/MatMul_3:product:0lstm_cell_3/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
lstm_cell_3/mul_4Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
lstm_cell_3/mul_5Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
lstm_cell_3/mul_6Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
lstm_cell_3/mul_7Mulzeros:output:0 lstm_cell_3/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/ReadVariableOpReadVariableOp#lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0p
lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   r
!lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
lstm_cell_3/strided_sliceStridedSlice"lstm_cell_3/ReadVariableOp:value:0(lstm_cell_3/strided_slice/stack:output:0*lstm_cell_3/strided_slice/stack_1:output:0*lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
lstm_cell_3/MatMul_4MatMullstm_cell_3/mul_4:z:0"lstm_cell_3/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/addAddV2lstm_cell_3/BiasAdd:output:0lstm_cell_3/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
lstm_cell_3/SigmoidSigmoidlstm_cell_3/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/ReadVariableOp_1ReadVariableOp#lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0r
!lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   t
#lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ±
lstm_cell_3/strided_slice_1StridedSlice$lstm_cell_3/ReadVariableOp_1:value:0*lstm_cell_3/strided_slice_1/stack:output:0,lstm_cell_3/strided_slice_1/stack_1:output:0,lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
lstm_cell_3/MatMul_5MatMullstm_cell_3/mul_5:z:0$lstm_cell_3/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/add_1AddV2lstm_cell_3/BiasAdd_1:output:0lstm_cell_3/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
lstm_cell_3/mul_8Mullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/ReadVariableOp_2ReadVariableOp#lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0r
!lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    À   t
#lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ±
lstm_cell_3/strided_slice_2StridedSlice$lstm_cell_3/ReadVariableOp_2:value:0*lstm_cell_3/strided_slice_2/stack:output:0,lstm_cell_3/strided_slice_2/stack_1:output:0,lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
lstm_cell_3/MatMul_6MatMullstm_cell_3/mul_6:z:0$lstm_cell_3/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/add_2AddV2lstm_cell_3/BiasAdd_2:output:0lstm_cell_3/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
lstm_cell_3/TanhTanhlstm_cell_3/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
lstm_cell_3/mul_9Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
lstm_cell_3/add_3AddV2lstm_cell_3/mul_8:z:0lstm_cell_3/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/ReadVariableOp_3ReadVariableOp#lstm_cell_3_readvariableop_resource*
_output_shapes
:	@*
dtype0r
!lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    À   t
#lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ±
lstm_cell_3/strided_slice_3StridedSlice$lstm_cell_3/ReadVariableOp_3:value:0*lstm_cell_3/strided_slice_3/stack:output:0,lstm_cell_3/strided_slice_3/stack_1:output:0,lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask
lstm_cell_3/MatMul_7MatMullstm_cell_3/mul_7:z:0$lstm_cell_3/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lstm_cell_3/add_4AddV2lstm_cell_3/BiasAdd_3:output:0lstm_cell_3/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
lstm_cell_3/Tanh_1Tanhlstm_cell_3/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
lstm_cell_3/mul_10Mullstm_cell_3/Sigmoid_2:y:0lstm_cell_3/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ó
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_3_split_readvariableop_resource+lstm_cell_3_split_1_readvariableop_resource#lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_24653*
condR
while_cond_24652*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@   Ã
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:úÿÿÿÿÿÿÿÿÿ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿú@[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^lstm_cell_3/ReadVariableOp^lstm_cell_3/ReadVariableOp_1^lstm_cell_3/ReadVariableOp_2^lstm_cell_3/ReadVariableOp_3!^lstm_cell_3/split/ReadVariableOp#^lstm_cell_3/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿúd: : : 28
lstm_cell_3/ReadVariableOplstm_cell_3/ReadVariableOp2<
lstm_cell_3/ReadVariableOp_1lstm_cell_3/ReadVariableOp_12<
lstm_cell_3/ReadVariableOp_2lstm_cell_3/ReadVariableOp_22<
lstm_cell_3/ReadVariableOp_3lstm_cell_3/ReadVariableOp_32D
 lstm_cell_3/split/ReadVariableOp lstm_cell_3/split/ReadVariableOp2H
"lstm_cell_3/split_1/ReadVariableOp"lstm_cell_3/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿúd
 
_user_specified_nameinputs
ç
ô
+__inference_lstm_cell_3_layer_call_fn_25215

inputs
states_0
states_1
unknown:	d
	unknown_0:	
	unknown_1:	@
identity

identity_1

identity_2¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_21866o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
states/1


ó
B__inference_dense_4_layer_call_and_return_conditional_losses_25178

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
½C
Ê
__inference__traced_save_25576
file_prefix5
1savev2_embedding_4_embeddings_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop8
4savev2_lstm_2_lstm_cell_3_kernel_read_readvariableopB
>savev2_lstm_2_lstm_cell_3_recurrent_kernel_read_readvariableop6
2savev2_lstm_2_lstm_cell_3_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableop?
;savev2_adam_lstm_2_lstm_cell_3_kernel_m_read_readvariableopI
Esavev2_adam_lstm_2_lstm_cell_3_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_2_lstm_cell_3_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableop?
;savev2_adam_lstm_2_lstm_cell_3_kernel_v_read_readvariableopI
Esavev2_adam_lstm_2_lstm_cell_3_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_2_lstm_cell_3_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*°
value¦B£ B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH­
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ²
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_4_embeddings_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop4savev2_lstm_2_lstm_cell_3_kernel_read_readvariableop>savev2_lstm_2_lstm_cell_3_recurrent_kernel_read_readvariableop2savev2_lstm_2_lstm_cell_3_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop;savev2_adam_lstm_2_lstm_cell_3_kernel_m_read_readvariableopEsavev2_adam_lstm_2_lstm_cell_3_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_2_lstm_cell_3_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableop;savev2_adam_lstm_2_lstm_cell_3_kernel_v_read_readvariableopEsavev2_adam_lstm_2_lstm_cell_3_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_2_lstm_cell_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*í
_input_shapesÛ
Ø: :	Äd:@ : : :: : : : : :	d:	@:: : : : :@ : : ::	d:	@::@ : : ::	d:	@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	Äd:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :%!

_output_shapes
:	d:%!

_output_shapes
:	@:!

_output_shapes	
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::%!

_output_shapes
:	d:%!

_output_shapes
:	@:!

_output_shapes	
::$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::%!

_output_shapes
:	d:%!

_output_shapes
:	@:!

_output_shapes	
:: 

_output_shapes
: "ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¿
serving_default«
P
embedding_4_input;
#serving_default_embedding_4_input:0ÿÿÿÿÿÿÿÿÿú;
dense_50
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
µ

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Ú
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
»

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
»

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses"
_tf_keras_layer
Ñ
.iter

/beta_1

0beta_2
	1decay
2learning_ratemimj&mk'ml3mm4mn5movpvq&vr'vs3vt4vu5vv"
	optimizer
X
0
31
42
53
4
5
&6
'7"
trackable_list_wrapper
Q
30
41
52
3
4
&5
'6"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
þ2û
,__inference_sequential_4_layer_call_fn_22588
,__inference_sequential_4_layer_call_fn_23171
,__inference_sequential_4_layer_call_fn_23192
,__inference_sequential_4_layer_call_fn_23096À
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
ê2ç
G__inference_sequential_4_layer_call_and_return_conditional_losses_23455
G__inference_sequential_4_layer_call_and_return_conditional_losses_23846
G__inference_sequential_4_layer_call_and_return_conditional_losses_23120
G__inference_sequential_4_layer_call_and_return_conditional_losses_23144À
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
ÕBÒ
 __inference__wrapped_model_21749embedding_4_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
;serving_default"
signature_map
):'	Äd2embedding_4/embeddings
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_embedding_4_layer_call_fn_23876¢
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
ð2í
F__inference_embedding_4_layer_call_and_return_conditional_losses_23886¢
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
ø
A
state_size

3kernel
4recurrent_kernel
5bias
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F_random_generator
G__call__
*H&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
30
41
52"
trackable_list_wrapper
5
30
41
52"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

Istates
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
û2ø
&__inference_lstm_2_layer_call_fn_23897
&__inference_lstm_2_layer_call_fn_23908
&__inference_lstm_2_layer_call_fn_23919
&__inference_lstm_2_layer_call_fn_23930Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ç2ä
A__inference_lstm_2_layer_call_and_return_conditional_losses_24173
A__inference_lstm_2_layer_call_and_return_conditional_losses_24544
A__inference_lstm_2_layer_call_and_return_conditional_losses_24787
A__inference_lstm_2_layer_call_and_return_conditional_losses_25158Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 :@ 2dense_4/kernel
: 2dense_4/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_dense_4_layer_call_fn_25167¢
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
ì2é
B__inference_dense_4_layer_call_and_return_conditional_losses_25178¢
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
 : 2dense_5/kernel
:2dense_5/bias
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_dense_5_layer_call_fn_25187¢
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
ì2é
B__inference_dense_5_layer_call_and_return_conditional_losses_25198¢
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
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
,:*	d2lstm_2/lstm_cell_3/kernel
6:4	@2#lstm_2/lstm_cell_3/recurrent_kernel
&:$2lstm_2/lstm_cell_3/bias
'
0"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÔBÑ
#__inference_signature_wrapper_23869embedding_4_input"
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
'
0"
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
5
30
41
52"
trackable_list_wrapper
5
30
41
52"
trackable_list_wrapper
 "
trackable_list_wrapper
­
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
+__inference_lstm_cell_3_layer_call_fn_25215
+__inference_lstm_cell_3_layer_call_fn_25232¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

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
Ô2Ñ
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_25314
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_25460¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
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
N
	`total
	acount
b	variables
c	keras_api"
_tf_keras_metric
^
	dtotal
	ecount
f
_fn_kwargs
g	variables
h	keras_api"
_tf_keras_metric
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
:  (2total
:  (2count
.
`0
a1"
trackable_list_wrapper
-
b	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
d0
e1"
trackable_list_wrapper
-
g	variables"
_generic_user_object
%:#@ 2Adam/dense_4/kernel/m
: 2Adam/dense_4/bias/m
%:# 2Adam/dense_5/kernel/m
:2Adam/dense_5/bias/m
1:/	d2 Adam/lstm_2/lstm_cell_3/kernel/m
;:9	@2*Adam/lstm_2/lstm_cell_3/recurrent_kernel/m
+:)2Adam/lstm_2/lstm_cell_3/bias/m
%:#@ 2Adam/dense_4/kernel/v
: 2Adam/dense_4/bias/v
%:# 2Adam/dense_5/kernel/v
:2Adam/dense_5/bias/v
1:/	d2 Adam/lstm_2/lstm_cell_3/kernel/v
;:9	@2*Adam/lstm_2/lstm_cell_3/recurrent_kernel/v
+:)2Adam/lstm_2/lstm_cell_3/bias/v
 __inference__wrapped_model_21749z354&';¢8
1¢.
,)
embedding_4_inputÿÿÿÿÿÿÿÿÿú
ª "1ª.
,
dense_5!
dense_5ÿÿÿÿÿÿÿÿÿ¢
B__inference_dense_4_layer_call_and_return_conditional_losses_25178\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 z
'__inference_dense_4_layer_call_fn_25167O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ ¢
B__inference_dense_5_layer_call_and_return_conditional_losses_25198\&'/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
'__inference_dense_5_layer_call_fn_25187O&'/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ«
F__inference_embedding_4_layer_call_and_return_conditional_losses_23886a0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿú
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿúd
 
+__inference_embedding_4_layer_call_fn_23876T0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿú
ª "ÿÿÿÿÿÿÿÿÿúdÂ
A__inference_lstm_2_layer_call_and_return_conditional_losses_24173}354O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 Â
A__inference_lstm_2_layer_call_and_return_conditional_losses_24544}354O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ³
A__inference_lstm_2_layer_call_and_return_conditional_losses_24787n354@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿúd

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ³
A__inference_lstm_2_layer_call_and_return_conditional_losses_25158n354@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿúd

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
&__inference_lstm_2_layer_call_fn_23897p354O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ@
&__inference_lstm_2_layer_call_fn_23908p354O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ@
&__inference_lstm_2_layer_call_fn_23919a354@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿúd

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ@
&__inference_lstm_2_layer_call_fn_23930a354@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿúd

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ@È
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_25314ý354¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿd
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ@
"
states/1ÿÿÿÿÿÿÿÿÿ@
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ@
EB

0/1/0ÿÿÿÿÿÿÿÿÿ@

0/1/1ÿÿÿÿÿÿÿÿÿ@
 È
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_25460ý354¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿd
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ@
"
states/1ÿÿÿÿÿÿÿÿÿ@
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ@
EB

0/1/0ÿÿÿÿÿÿÿÿÿ@

0/1/1ÿÿÿÿÿÿÿÿÿ@
 
+__inference_lstm_cell_3_layer_call_fn_25215í354¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿd
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ@
"
states/1ÿÿÿÿÿÿÿÿÿ@
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ@
A>

1/0ÿÿÿÿÿÿÿÿÿ@

1/1ÿÿÿÿÿÿÿÿÿ@
+__inference_lstm_cell_3_layer_call_fn_25232í354¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿd
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ@
"
states/1ÿÿÿÿÿÿÿÿÿ@
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ@
A>

1/0ÿÿÿÿÿÿÿÿÿ@

1/1ÿÿÿÿÿÿÿÿÿ@Á
G__inference_sequential_4_layer_call_and_return_conditional_losses_23120v354&'C¢@
9¢6
,)
embedding_4_inputÿÿÿÿÿÿÿÿÿú
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Á
G__inference_sequential_4_layer_call_and_return_conditional_losses_23144v354&'C¢@
9¢6
,)
embedding_4_inputÿÿÿÿÿÿÿÿÿú
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¶
G__inference_sequential_4_layer_call_and_return_conditional_losses_23455k354&'8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿú
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¶
G__inference_sequential_4_layer_call_and_return_conditional_losses_23846k354&'8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿú
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_sequential_4_layer_call_fn_22588i354&'C¢@
9¢6
,)
embedding_4_inputÿÿÿÿÿÿÿÿÿú
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_4_layer_call_fn_23096i354&'C¢@
9¢6
,)
embedding_4_inputÿÿÿÿÿÿÿÿÿú
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_4_layer_call_fn_23171^354&'8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿú
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_4_layer_call_fn_23192^354&'8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿú
p

 
ª "ÿÿÿÿÿÿÿÿÿ·
#__inference_signature_wrapper_23869354&'P¢M
¢ 
FªC
A
embedding_4_input,)
embedding_4_inputÿÿÿÿÿÿÿÿÿú"1ª.
,
dense_5!
dense_5ÿÿÿÿÿÿÿÿÿ