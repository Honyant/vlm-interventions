К,
8ж7
.
Abs
x"T
y"T"
Ttype:

2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
s
	AssignAdd
ref"T

value"T

output_ref"T" 
Ttype:
2	"
use_lockingbool( 
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
.
Atanh
x"T
y"T"
Ttype:

2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(
,
Exp
x"T
y"T"
Ttype:

2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
A
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
?
FloorMod
x"T
y"T
z"T"
Ttype:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
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
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

Min

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
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
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
8
Pow
x"T
y"T
z"T"
Ttype:
2
	

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
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
7

Reciprocal
x"T
y"T"
Ttype:
2
	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
р
ResourceApplyAdam
var
m
v
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
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
A
SelectV2
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
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
7
Square
x"T
y"T"
Ttype:
2	
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
ї
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 
9
VarIsInitializedOp
resource
is_initialized

s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e5сх)

,default_policy/global_step/Initializer/zerosConst*-
_class#
!loc:@default_policy/global_step*
_output_shapes
: *
dtype0	*
value	B	 R 
­
default_policy/global_step
VariableV2*-
_class#
!loc:@default_policy/global_step*
_output_shapes
: *
	container *
dtype0	*
shape: *
shared_name 

!default_policy/global_step/AssignAssigndefault_policy/global_step,default_policy/global_step/Initializer/zeros*
T0	*-
_class#
!loc:@default_policy/global_step*&
 _has_manual_control_dependencies(*
_output_shapes
: *
use_locking(*
validate_shape(

default_policy/global_step/readIdentitydefault_policy/global_step*
T0	*-
_class#
!loc:@default_policy/global_step*
_output_shapes
: 

default_policy/observationPlaceholder*(
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
x
default_policy/actionPlaceholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
c
!default_policy/is_exploring/inputConst*
_output_shapes
: *
dtype0
*
value	B
 Z

default_policy/is_exploringPlaceholderWithDefault!default_policy/is_exploring/input*
_output_shapes
: *
dtype0
*
shape: 
X
default_policy/timestepPlaceholder*
_output_shapes
: *
dtype0*
shape: 
b
 default_policy/is_training/inputConst*
_output_shapes
: *
dtype0
*
value	B
 Z 

default_policy/is_trainingPlaceholderWithDefault default_policy/is_training/input*
_output_shapes
: *
dtype0
*
shape: 
r
default_policy/seq_lensPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

default_policy/observationsPlaceholder*(
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
Ш	
1default_policy/value_out/kernel/Initializer/ConstConst*2
_class(
&$loc:@default_policy/value_out/kernel*
_output_shapes
:	*
dtype0*Љ
valueB	"2КДZ9Ts7:
Ї ЙY!:n Ѓ9ЛТ9?*К1ёИџH"9ПЙx9жГЙЙOі:ж9uхИDЉЙ-е#:ЊjЙFэ1КАЪ:бEu:NL ЙКЭлъ8:ш9e X:щOЙЛ8|9ueКВЌ№ЙLц:wЫю9ЙE=:}КXЮWК@DК/>­Й{дЙ/і6/
УЙёЪN:ЁоК`rКcc9Ї\Кhўi:%=КюЎЙ:КаЙК58TЁ9Ї.	КЫF:BЉJИчКЩKКЖЪ=7Ы]9hѕ	7ЏКд"&КWУ9Cъ9Ћb#:YнcКђwК#ќh9V%SЙъOлЙђкЁ:_=ЏЙ*Ью9 ђOКЏНИ9fЛ/Кл9Hк9nCyКНћ:гї9ќC9ќ8цџ7U/ИЙже>КйЕКl№j9H:UнoЙЛ:Эѕ5*AИ1!6;}ыИ	лХК[КШ4КЭьК Љ7eЙ?ЙpЈ8З1b7ЪљI9хkяЙ>BNКЖ0r8їгЙKБ<КLhЙј8JК§\|:вЃм9zФКдЋИuК~бЂ:D^К§.К@ыЙ^cК§:EEЗU:зМё9bJnК)ѓ=Кћ<N:V:єЊAК{+ШИ.мoКQ8cЗэ9Й9ы"aКфЫЙј9wiЙSЅ:FzZЙв6ФЙ.йђ9bѕ8%У]ЙDмЖ:HвИф`9}в+9KЙБ6Кp: ЪхЙ.:-Йг8ЕЖйЌЙЎ я9YКА:LJfКRШЅ8:zO К6{JИ+?9	К9№ёйЙЁ ИT:kІ9p&ЗPЙ
$XКБЃЙІ	Й9g9.КsЫD: Й &рЙ-МЙлъ9тБ9@H:3CЃ9П;::ЦHЅ:ЙдЮ9wоИVй!:ЮъОЙХz:TР9}OКfbМ9gЙП8,)P:>Х\:`КЂ79ЬWЙЈЯЇ9Љ5nЙа9vzcКчЛ9§ЧКїО9V	ЫИwuE:V:ЇЁвЙад	;my9CP9м$FЙrЊbКIжЦЙ	И0:tК|%.КАЙ(^НЙ^Кч`8їЬПЙ3::ДК.]Й=ЦаИA:ЗwЧЙ]љ.Й6:И8:+pп9ЅELКе3r:yхИ*boЙ1|9g:tВ9lХПЗxL:зъЖ:'~КhT8
ї
default_policy/value_out/kernelVarHandleOp*2
_class(
&$loc:@default_policy/value_out/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:	*0
shared_name!default_policy/value_out/kernel

@default_policy/value_out/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/value_out/kernel*
_output_shapes
: 
й
&default_policy/value_out/kernel/AssignAssignVariableOpdefault_policy/value_out/kernel1default_policy/value_out/kernel/Initializer/Const*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

3default_policy/value_out/kernel/Read/ReadVariableOpReadVariableOpdefault_policy/value_out/kernel*
_output_shapes
:	*
dtype0
Ў
/default_policy/value_out/bias/Initializer/zerosConst*0
_class&
$"loc:@default_policy/value_out/bias*
_output_shapes
:*
dtype0*
valueB*    
ь
default_policy/value_out/biasVarHandleOp*0
_class&
$"loc:@default_policy/value_out/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*.
shared_namedefault_policy/value_out/bias

>default_policy/value_out/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/value_out/bias*
_output_shapes
: 
г
$default_policy/value_out/bias/AssignAssignVariableOpdefault_policy/value_out/bias/default_policy/value_out/bias/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

1default_policy/value_out/bias/Read/ReadVariableOpReadVariableOpdefault_policy/value_out/bias*
_output_shapes
:*
dtype0

.default_policy/value_out/MatMul/ReadVariableOpReadVariableOpdefault_policy/value_out/kernel*
_output_shapes
:	*
dtype0
Ю
default_policy/value_out/MatMulMatMuldefault_policy/observations.default_policy/value_out/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

/default_policy/value_out/BiasAdd/ReadVariableOpReadVariableOpdefault_policy/value_out/bias*
_output_shapes
:*
dtype0
Ц
 default_policy/value_out/BiasAddBiasAdddefault_policy/value_out/MatMul/default_policy/value_out/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
}
default_policy/model_outPlaceholder*(
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
й
Jdefault_policy/sequential/action_1/kernel/Initializer/random_uniform/shapeConst*<
_class2
0.loc:@default_policy/sequential/action_1/kernel*
_output_shapes
:*
dtype0*
valueB"     
Ы
Hdefault_policy/sequential/action_1/kernel/Initializer/random_uniform/minConst*<
_class2
0.loc:@default_policy/sequential/action_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *KнН
Ы
Hdefault_policy/sequential/action_1/kernel/Initializer/random_uniform/maxConst*<
_class2
0.loc:@default_policy/sequential/action_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *Kн=
О
Rdefault_policy/sequential/action_1/kernel/Initializer/random_uniform/RandomUniformRandomUniformJdefault_policy/sequential/action_1/kernel/Initializer/random_uniform/shape*
T0*<
_class2
0.loc:@default_policy/sequential/action_1/kernel* 
_output_shapes
:
*
dtype0*

seedd*
seed2 
Т
Hdefault_policy/sequential/action_1/kernel/Initializer/random_uniform/subSubHdefault_policy/sequential/action_1/kernel/Initializer/random_uniform/maxHdefault_policy/sequential/action_1/kernel/Initializer/random_uniform/min*
T0*<
_class2
0.loc:@default_policy/sequential/action_1/kernel*
_output_shapes
: 
ж
Hdefault_policy/sequential/action_1/kernel/Initializer/random_uniform/mulMulRdefault_policy/sequential/action_1/kernel/Initializer/random_uniform/RandomUniformHdefault_policy/sequential/action_1/kernel/Initializer/random_uniform/sub*
T0*<
_class2
0.loc:@default_policy/sequential/action_1/kernel* 
_output_shapes
:

Ъ
Ddefault_policy/sequential/action_1/kernel/Initializer/random_uniformAddV2Hdefault_policy/sequential/action_1/kernel/Initializer/random_uniform/mulHdefault_policy/sequential/action_1/kernel/Initializer/random_uniform/min*
T0*<
_class2
0.loc:@default_policy/sequential/action_1/kernel* 
_output_shapes
:


)default_policy/sequential/action_1/kernelVarHandleOp*<
_class2
0.loc:@default_policy/sequential/action_1/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:
*:
shared_name+)default_policy/sequential/action_1/kernel
Ѓ
Jdefault_policy/sequential/action_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp)default_policy/sequential/action_1/kernel*
_output_shapes
: 

0default_policy/sequential/action_1/kernel/AssignAssignVariableOp)default_policy/sequential/action_1/kernelDdefault_policy/sequential/action_1/kernel/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Љ
=default_policy/sequential/action_1/kernel/Read/ReadVariableOpReadVariableOp)default_policy/sequential/action_1/kernel* 
_output_shapes
:
*
dtype0
Ф
9default_policy/sequential/action_1/bias/Initializer/zerosConst*:
_class0
.,loc:@default_policy/sequential/action_1/bias*
_output_shapes	
:*
dtype0*
valueB*    

'default_policy/sequential/action_1/biasVarHandleOp*:
_class0
.,loc:@default_policy/sequential/action_1/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*8
shared_name)'default_policy/sequential/action_1/bias

Hdefault_policy/sequential/action_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp'default_policy/sequential/action_1/bias*
_output_shapes
: 
ё
.default_policy/sequential/action_1/bias/AssignAssignVariableOp'default_policy/sequential/action_1/bias9default_policy/sequential/action_1/bias/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
 
;default_policy/sequential/action_1/bias/Read/ReadVariableOpReadVariableOp'default_policy/sequential/action_1/bias*
_output_shapes	
:*
dtype0
Є
8default_policy/sequential/action_1/MatMul/ReadVariableOpReadVariableOp)default_policy/sequential/action_1/kernel* 
_output_shapes
:
*
dtype0
р
)default_policy/sequential/action_1/MatMulMatMuldefault_policy/model_out8default_policy/sequential/action_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

9default_policy/sequential/action_1/BiasAdd/ReadVariableOpReadVariableOp'default_policy/sequential/action_1/bias*
_output_shapes	
:*
dtype0
х
*default_policy/sequential/action_1/BiasAddBiasAdd)default_policy/sequential/action_1/MatMul9default_policy/sequential/action_1/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC

'default_policy/sequential/action_1/ReluRelu*default_policy/sequential/action_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
й
Jdefault_policy/sequential/action_2/kernel/Initializer/random_uniform/shapeConst*<
_class2
0.loc:@default_policy/sequential/action_2/kernel*
_output_shapes
:*
dtype0*
valueB"      
Ы
Hdefault_policy/sequential/action_2/kernel/Initializer/random_uniform/minConst*<
_class2
0.loc:@default_policy/sequential/action_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *зГнН
Ы
Hdefault_policy/sequential/action_2/kernel/Initializer/random_uniform/maxConst*<
_class2
0.loc:@default_policy/sequential/action_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *зГн=
О
Rdefault_policy/sequential/action_2/kernel/Initializer/random_uniform/RandomUniformRandomUniformJdefault_policy/sequential/action_2/kernel/Initializer/random_uniform/shape*
T0*<
_class2
0.loc:@default_policy/sequential/action_2/kernel* 
_output_shapes
:
*
dtype0*

seedd*
seed2
Т
Hdefault_policy/sequential/action_2/kernel/Initializer/random_uniform/subSubHdefault_policy/sequential/action_2/kernel/Initializer/random_uniform/maxHdefault_policy/sequential/action_2/kernel/Initializer/random_uniform/min*
T0*<
_class2
0.loc:@default_policy/sequential/action_2/kernel*
_output_shapes
: 
ж
Hdefault_policy/sequential/action_2/kernel/Initializer/random_uniform/mulMulRdefault_policy/sequential/action_2/kernel/Initializer/random_uniform/RandomUniformHdefault_policy/sequential/action_2/kernel/Initializer/random_uniform/sub*
T0*<
_class2
0.loc:@default_policy/sequential/action_2/kernel* 
_output_shapes
:

Ъ
Ddefault_policy/sequential/action_2/kernel/Initializer/random_uniformAddV2Hdefault_policy/sequential/action_2/kernel/Initializer/random_uniform/mulHdefault_policy/sequential/action_2/kernel/Initializer/random_uniform/min*
T0*<
_class2
0.loc:@default_policy/sequential/action_2/kernel* 
_output_shapes
:


)default_policy/sequential/action_2/kernelVarHandleOp*<
_class2
0.loc:@default_policy/sequential/action_2/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:
*:
shared_name+)default_policy/sequential/action_2/kernel
Ѓ
Jdefault_policy/sequential/action_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp)default_policy/sequential/action_2/kernel*
_output_shapes
: 

0default_policy/sequential/action_2/kernel/AssignAssignVariableOp)default_policy/sequential/action_2/kernelDdefault_policy/sequential/action_2/kernel/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Љ
=default_policy/sequential/action_2/kernel/Read/ReadVariableOpReadVariableOp)default_policy/sequential/action_2/kernel* 
_output_shapes
:
*
dtype0
Ф
9default_policy/sequential/action_2/bias/Initializer/zerosConst*:
_class0
.,loc:@default_policy/sequential/action_2/bias*
_output_shapes	
:*
dtype0*
valueB*    

'default_policy/sequential/action_2/biasVarHandleOp*:
_class0
.,loc:@default_policy/sequential/action_2/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*8
shared_name)'default_policy/sequential/action_2/bias

Hdefault_policy/sequential/action_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp'default_policy/sequential/action_2/bias*
_output_shapes
: 
ё
.default_policy/sequential/action_2/bias/AssignAssignVariableOp'default_policy/sequential/action_2/bias9default_policy/sequential/action_2/bias/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
 
;default_policy/sequential/action_2/bias/Read/ReadVariableOpReadVariableOp'default_policy/sequential/action_2/bias*
_output_shapes	
:*
dtype0
Є
8default_policy/sequential/action_2/MatMul/ReadVariableOpReadVariableOp)default_policy/sequential/action_2/kernel* 
_output_shapes
:
*
dtype0
я
)default_policy/sequential/action_2/MatMulMatMul'default_policy/sequential/action_1/Relu8default_policy/sequential/action_2/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

9default_policy/sequential/action_2/BiasAdd/ReadVariableOpReadVariableOp'default_policy/sequential/action_2/bias*
_output_shapes	
:*
dtype0
х
*default_policy/sequential/action_2/BiasAddBiasAdd)default_policy/sequential/action_2/MatMul9default_policy/sequential/action_2/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC

'default_policy/sequential/action_2/ReluRelu*default_policy/sequential/action_2/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
н
Ldefault_policy/sequential/action_out/kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@default_policy/sequential/action_out/kernel*
_output_shapes
:*
dtype0*
valueB"      
Я
Jdefault_policy/sequential/action_out/kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@default_policy/sequential/action_out/kernel*
_output_shapes
: *
dtype0*
valueB
 *О
Я
Jdefault_policy/sequential/action_out/kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@default_policy/sequential/action_out/kernel*
_output_shapes
: *
dtype0*
valueB
 *>
У
Tdefault_policy/sequential/action_out/kernel/Initializer/random_uniform/RandomUniformRandomUniformLdefault_policy/sequential/action_out/kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@default_policy/sequential/action_out/kernel*
_output_shapes
:	*
dtype0*

seedd*
seed2
Ъ
Jdefault_policy/sequential/action_out/kernel/Initializer/random_uniform/subSubJdefault_policy/sequential/action_out/kernel/Initializer/random_uniform/maxJdefault_policy/sequential/action_out/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@default_policy/sequential/action_out/kernel*
_output_shapes
: 
н
Jdefault_policy/sequential/action_out/kernel/Initializer/random_uniform/mulMulTdefault_policy/sequential/action_out/kernel/Initializer/random_uniform/RandomUniformJdefault_policy/sequential/action_out/kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@default_policy/sequential/action_out/kernel*
_output_shapes
:	
б
Fdefault_policy/sequential/action_out/kernel/Initializer/random_uniformAddV2Jdefault_policy/sequential/action_out/kernel/Initializer/random_uniform/mulJdefault_policy/sequential/action_out/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@default_policy/sequential/action_out/kernel*
_output_shapes
:	

+default_policy/sequential/action_out/kernelVarHandleOp*>
_class4
20loc:@default_policy/sequential/action_out/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:	*<
shared_name-+default_policy/sequential/action_out/kernel
Ї
Ldefault_policy/sequential/action_out/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+default_policy/sequential/action_out/kernel*
_output_shapes
: 

2default_policy/sequential/action_out/kernel/AssignAssignVariableOp+default_policy/sequential/action_out/kernelFdefault_policy/sequential/action_out/kernel/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Ќ
?default_policy/sequential/action_out/kernel/Read/ReadVariableOpReadVariableOp+default_policy/sequential/action_out/kernel*
_output_shapes
:	*
dtype0
Ц
;default_policy/sequential/action_out/bias/Initializer/zerosConst*<
_class2
0.loc:@default_policy/sequential/action_out/bias*
_output_shapes
:*
dtype0*
valueB*    

)default_policy/sequential/action_out/biasVarHandleOp*<
_class2
0.loc:@default_policy/sequential/action_out/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*:
shared_name+)default_policy/sequential/action_out/bias
Ѓ
Jdefault_policy/sequential/action_out/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp)default_policy/sequential/action_out/bias*
_output_shapes
: 
ї
0default_policy/sequential/action_out/bias/AssignAssignVariableOp)default_policy/sequential/action_out/bias;default_policy/sequential/action_out/bias/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Ѓ
=default_policy/sequential/action_out/bias/Read/ReadVariableOpReadVariableOp)default_policy/sequential/action_out/bias*
_output_shapes
:*
dtype0
Ї
:default_policy/sequential/action_out/MatMul/ReadVariableOpReadVariableOp+default_policy/sequential/action_out/kernel*
_output_shapes
:	*
dtype0
ђ
+default_policy/sequential/action_out/MatMulMatMul'default_policy/sequential/action_2/Relu:default_policy/sequential/action_out/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ё
;default_policy/sequential/action_out/BiasAdd/ReadVariableOpReadVariableOp)default_policy/sequential/action_out/bias*
_output_shapes
:*
dtype0
ъ
,default_policy/sequential/action_out/BiasAddBiasAdd+default_policy/sequential/action_out/MatMul;default_policy/sequential/action_out/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
y
default_policy/actionsPlaceholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
u
3default_policy/sequential_1/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
щ
.default_policy/sequential_1/concatenate/concatConcatV2default_policy/model_outdefault_policy/actions3default_policy/sequential_1/concatenate/concat/axis*
N*
T0*

Tidx0*(
_output_shapes
:џџџџџџџџџ
с
Ndefault_policy/sequential_1/q_hidden_0/kernel/Initializer/random_uniform/shapeConst*@
_class6
42loc:@default_policy/sequential_1/q_hidden_0/kernel*
_output_shapes
:*
dtype0*
valueB"     
г
Ldefault_policy/sequential_1/q_hidden_0/kernel/Initializer/random_uniform/minConst*@
_class6
42loc:@default_policy/sequential_1/q_hidden_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *К мН
г
Ldefault_policy/sequential_1/q_hidden_0/kernel/Initializer/random_uniform/maxConst*@
_class6
42loc:@default_policy/sequential_1/q_hidden_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *К м=
Ъ
Vdefault_policy/sequential_1/q_hidden_0/kernel/Initializer/random_uniform/RandomUniformRandomUniformNdefault_policy/sequential_1/q_hidden_0/kernel/Initializer/random_uniform/shape*
T0*@
_class6
42loc:@default_policy/sequential_1/q_hidden_0/kernel* 
_output_shapes
:
*
dtype0*

seedd*
seed2
в
Ldefault_policy/sequential_1/q_hidden_0/kernel/Initializer/random_uniform/subSubLdefault_policy/sequential_1/q_hidden_0/kernel/Initializer/random_uniform/maxLdefault_policy/sequential_1/q_hidden_0/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@default_policy/sequential_1/q_hidden_0/kernel*
_output_shapes
: 
ц
Ldefault_policy/sequential_1/q_hidden_0/kernel/Initializer/random_uniform/mulMulVdefault_policy/sequential_1/q_hidden_0/kernel/Initializer/random_uniform/RandomUniformLdefault_policy/sequential_1/q_hidden_0/kernel/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@default_policy/sequential_1/q_hidden_0/kernel* 
_output_shapes
:

к
Hdefault_policy/sequential_1/q_hidden_0/kernel/Initializer/random_uniformAddV2Ldefault_policy/sequential_1/q_hidden_0/kernel/Initializer/random_uniform/mulLdefault_policy/sequential_1/q_hidden_0/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@default_policy/sequential_1/q_hidden_0/kernel* 
_output_shapes
:

Ђ
-default_policy/sequential_1/q_hidden_0/kernelVarHandleOp*@
_class6
42loc:@default_policy/sequential_1/q_hidden_0/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:
*>
shared_name/-default_policy/sequential_1/q_hidden_0/kernel
Ћ
Ndefault_policy/sequential_1/q_hidden_0/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp-default_policy/sequential_1/q_hidden_0/kernel*
_output_shapes
: 

4default_policy/sequential_1/q_hidden_0/kernel/AssignAssignVariableOp-default_policy/sequential_1/q_hidden_0/kernelHdefault_policy/sequential_1/q_hidden_0/kernel/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Б
Adefault_policy/sequential_1/q_hidden_0/kernel/Read/ReadVariableOpReadVariableOp-default_policy/sequential_1/q_hidden_0/kernel* 
_output_shapes
:
*
dtype0
Ь
=default_policy/sequential_1/q_hidden_0/bias/Initializer/zerosConst*>
_class4
20loc:@default_policy/sequential_1/q_hidden_0/bias*
_output_shapes	
:*
dtype0*
valueB*    

+default_policy/sequential_1/q_hidden_0/biasVarHandleOp*>
_class4
20loc:@default_policy/sequential_1/q_hidden_0/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*<
shared_name-+default_policy/sequential_1/q_hidden_0/bias
Ї
Ldefault_policy/sequential_1/q_hidden_0/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp+default_policy/sequential_1/q_hidden_0/bias*
_output_shapes
: 
§
2default_policy/sequential_1/q_hidden_0/bias/AssignAssignVariableOp+default_policy/sequential_1/q_hidden_0/bias=default_policy/sequential_1/q_hidden_0/bias/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Ј
?default_policy/sequential_1/q_hidden_0/bias/Read/ReadVariableOpReadVariableOp+default_policy/sequential_1/q_hidden_0/bias*
_output_shapes	
:*
dtype0
Ќ
<default_policy/sequential_1/q_hidden_0/MatMul/ReadVariableOpReadVariableOp-default_policy/sequential_1/q_hidden_0/kernel* 
_output_shapes
:
*
dtype0
ў
-default_policy/sequential_1/q_hidden_0/MatMulMatMul.default_policy/sequential_1/concatenate/concat<default_policy/sequential_1/q_hidden_0/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
І
=default_policy/sequential_1/q_hidden_0/BiasAdd/ReadVariableOpReadVariableOp+default_policy/sequential_1/q_hidden_0/bias*
_output_shapes	
:*
dtype0
ё
.default_policy/sequential_1/q_hidden_0/BiasAddBiasAdd-default_policy/sequential_1/q_hidden_0/MatMul=default_policy/sequential_1/q_hidden_0/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC

+default_policy/sequential_1/q_hidden_0/ReluRelu.default_policy/sequential_1/q_hidden_0/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
с
Ndefault_policy/sequential_1/q_hidden_1/kernel/Initializer/random_uniform/shapeConst*@
_class6
42loc:@default_policy/sequential_1/q_hidden_1/kernel*
_output_shapes
:*
dtype0*
valueB"      
г
Ldefault_policy/sequential_1/q_hidden_1/kernel/Initializer/random_uniform/minConst*@
_class6
42loc:@default_policy/sequential_1/q_hidden_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *зГнН
г
Ldefault_policy/sequential_1/q_hidden_1/kernel/Initializer/random_uniform/maxConst*@
_class6
42loc:@default_policy/sequential_1/q_hidden_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *зГн=
Ъ
Vdefault_policy/sequential_1/q_hidden_1/kernel/Initializer/random_uniform/RandomUniformRandomUniformNdefault_policy/sequential_1/q_hidden_1/kernel/Initializer/random_uniform/shape*
T0*@
_class6
42loc:@default_policy/sequential_1/q_hidden_1/kernel* 
_output_shapes
:
*
dtype0*

seedd*
seed2
в
Ldefault_policy/sequential_1/q_hidden_1/kernel/Initializer/random_uniform/subSubLdefault_policy/sequential_1/q_hidden_1/kernel/Initializer/random_uniform/maxLdefault_policy/sequential_1/q_hidden_1/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@default_policy/sequential_1/q_hidden_1/kernel*
_output_shapes
: 
ц
Ldefault_policy/sequential_1/q_hidden_1/kernel/Initializer/random_uniform/mulMulVdefault_policy/sequential_1/q_hidden_1/kernel/Initializer/random_uniform/RandomUniformLdefault_policy/sequential_1/q_hidden_1/kernel/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@default_policy/sequential_1/q_hidden_1/kernel* 
_output_shapes
:

к
Hdefault_policy/sequential_1/q_hidden_1/kernel/Initializer/random_uniformAddV2Ldefault_policy/sequential_1/q_hidden_1/kernel/Initializer/random_uniform/mulLdefault_policy/sequential_1/q_hidden_1/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@default_policy/sequential_1/q_hidden_1/kernel* 
_output_shapes
:

Ђ
-default_policy/sequential_1/q_hidden_1/kernelVarHandleOp*@
_class6
42loc:@default_policy/sequential_1/q_hidden_1/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:
*>
shared_name/-default_policy/sequential_1/q_hidden_1/kernel
Ћ
Ndefault_policy/sequential_1/q_hidden_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp-default_policy/sequential_1/q_hidden_1/kernel*
_output_shapes
: 

4default_policy/sequential_1/q_hidden_1/kernel/AssignAssignVariableOp-default_policy/sequential_1/q_hidden_1/kernelHdefault_policy/sequential_1/q_hidden_1/kernel/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Б
Adefault_policy/sequential_1/q_hidden_1/kernel/Read/ReadVariableOpReadVariableOp-default_policy/sequential_1/q_hidden_1/kernel* 
_output_shapes
:
*
dtype0
Ь
=default_policy/sequential_1/q_hidden_1/bias/Initializer/zerosConst*>
_class4
20loc:@default_policy/sequential_1/q_hidden_1/bias*
_output_shapes	
:*
dtype0*
valueB*    

+default_policy/sequential_1/q_hidden_1/biasVarHandleOp*>
_class4
20loc:@default_policy/sequential_1/q_hidden_1/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*<
shared_name-+default_policy/sequential_1/q_hidden_1/bias
Ї
Ldefault_policy/sequential_1/q_hidden_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp+default_policy/sequential_1/q_hidden_1/bias*
_output_shapes
: 
§
2default_policy/sequential_1/q_hidden_1/bias/AssignAssignVariableOp+default_policy/sequential_1/q_hidden_1/bias=default_policy/sequential_1/q_hidden_1/bias/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Ј
?default_policy/sequential_1/q_hidden_1/bias/Read/ReadVariableOpReadVariableOp+default_policy/sequential_1/q_hidden_1/bias*
_output_shapes	
:*
dtype0
Ќ
<default_policy/sequential_1/q_hidden_1/MatMul/ReadVariableOpReadVariableOp-default_policy/sequential_1/q_hidden_1/kernel* 
_output_shapes
:
*
dtype0
ћ
-default_policy/sequential_1/q_hidden_1/MatMulMatMul+default_policy/sequential_1/q_hidden_0/Relu<default_policy/sequential_1/q_hidden_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
І
=default_policy/sequential_1/q_hidden_1/BiasAdd/ReadVariableOpReadVariableOp+default_policy/sequential_1/q_hidden_1/bias*
_output_shapes	
:*
dtype0
ё
.default_policy/sequential_1/q_hidden_1/BiasAddBiasAdd-default_policy/sequential_1/q_hidden_1/MatMul=default_policy/sequential_1/q_hidden_1/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC

+default_policy/sequential_1/q_hidden_1/ReluRelu.default_policy/sequential_1/q_hidden_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
з
Idefault_policy/sequential_1/q_out/kernel/Initializer/random_uniform/shapeConst*;
_class1
/-loc:@default_policy/sequential_1/q_out/kernel*
_output_shapes
:*
dtype0*
valueB"      
Щ
Gdefault_policy/sequential_1/q_out/kernel/Initializer/random_uniform/minConst*;
_class1
/-loc:@default_policy/sequential_1/q_out/kernel*
_output_shapes
: *
dtype0*
valueB
 *IvО
Щ
Gdefault_policy/sequential_1/q_out/kernel/Initializer/random_uniform/maxConst*;
_class1
/-loc:@default_policy/sequential_1/q_out/kernel*
_output_shapes
: *
dtype0*
valueB
 *Iv>
К
Qdefault_policy/sequential_1/q_out/kernel/Initializer/random_uniform/RandomUniformRandomUniformIdefault_policy/sequential_1/q_out/kernel/Initializer/random_uniform/shape*
T0*;
_class1
/-loc:@default_policy/sequential_1/q_out/kernel*
_output_shapes
:	*
dtype0*

seedd*
seed2
О
Gdefault_policy/sequential_1/q_out/kernel/Initializer/random_uniform/subSubGdefault_policy/sequential_1/q_out/kernel/Initializer/random_uniform/maxGdefault_policy/sequential_1/q_out/kernel/Initializer/random_uniform/min*
T0*;
_class1
/-loc:@default_policy/sequential_1/q_out/kernel*
_output_shapes
: 
б
Gdefault_policy/sequential_1/q_out/kernel/Initializer/random_uniform/mulMulQdefault_policy/sequential_1/q_out/kernel/Initializer/random_uniform/RandomUniformGdefault_policy/sequential_1/q_out/kernel/Initializer/random_uniform/sub*
T0*;
_class1
/-loc:@default_policy/sequential_1/q_out/kernel*
_output_shapes
:	
Х
Cdefault_policy/sequential_1/q_out/kernel/Initializer/random_uniformAddV2Gdefault_policy/sequential_1/q_out/kernel/Initializer/random_uniform/mulGdefault_policy/sequential_1/q_out/kernel/Initializer/random_uniform/min*
T0*;
_class1
/-loc:@default_policy/sequential_1/q_out/kernel*
_output_shapes
:	

(default_policy/sequential_1/q_out/kernelVarHandleOp*;
_class1
/-loc:@default_policy/sequential_1/q_out/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:	*9
shared_name*(default_policy/sequential_1/q_out/kernel
Ё
Idefault_policy/sequential_1/q_out/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp(default_policy/sequential_1/q_out/kernel*
_output_shapes
: 
§
/default_policy/sequential_1/q_out/kernel/AssignAssignVariableOp(default_policy/sequential_1/q_out/kernelCdefault_policy/sequential_1/q_out/kernel/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
І
<default_policy/sequential_1/q_out/kernel/Read/ReadVariableOpReadVariableOp(default_policy/sequential_1/q_out/kernel*
_output_shapes
:	*
dtype0
Р
8default_policy/sequential_1/q_out/bias/Initializer/zerosConst*9
_class/
-+loc:@default_policy/sequential_1/q_out/bias*
_output_shapes
:*
dtype0*
valueB*    

&default_policy/sequential_1/q_out/biasVarHandleOp*9
_class/
-+loc:@default_policy/sequential_1/q_out/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*7
shared_name(&default_policy/sequential_1/q_out/bias

Gdefault_policy/sequential_1/q_out/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp&default_policy/sequential_1/q_out/bias*
_output_shapes
: 
ю
-default_policy/sequential_1/q_out/bias/AssignAssignVariableOp&default_policy/sequential_1/q_out/bias8default_policy/sequential_1/q_out/bias/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

:default_policy/sequential_1/q_out/bias/Read/ReadVariableOpReadVariableOp&default_policy/sequential_1/q_out/bias*
_output_shapes
:*
dtype0
Ё
7default_policy/sequential_1/q_out/MatMul/ReadVariableOpReadVariableOp(default_policy/sequential_1/q_out/kernel*
_output_shapes
:	*
dtype0
№
(default_policy/sequential_1/q_out/MatMulMatMul+default_policy/sequential_1/q_hidden_1/Relu7default_policy/sequential_1/q_out/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

8default_policy/sequential_1/q_out/BiasAdd/ReadVariableOpReadVariableOp&default_policy/sequential_1/q_out/bias*
_output_shapes
:*
dtype0
с
)default_policy/sequential_1/q_out/BiasAddBiasAdd(default_policy/sequential_1/q_out/MatMul8default_policy/sequential_1/q_out/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
w
5default_policy/sequential_2/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
э
0default_policy/sequential_2/concatenate_1/concatConcatV2default_policy/model_outdefault_policy/actions5default_policy/sequential_2/concatenate_1/concat/axis*
N*
T0*

Tidx0*(
_output_shapes
:џџџџџџџџџ
ы
Sdefault_policy/sequential_2/twin_q_hidden_0/kernel/Initializer/random_uniform/shapeConst*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_0/kernel*
_output_shapes
:*
dtype0*
valueB"     
н
Qdefault_policy/sequential_2/twin_q_hidden_0/kernel/Initializer/random_uniform/minConst*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *К мН
н
Qdefault_policy/sequential_2/twin_q_hidden_0/kernel/Initializer/random_uniform/maxConst*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *К м=
й
[default_policy/sequential_2/twin_q_hidden_0/kernel/Initializer/random_uniform/RandomUniformRandomUniformSdefault_policy/sequential_2/twin_q_hidden_0/kernel/Initializer/random_uniform/shape*
T0*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_0/kernel* 
_output_shapes
:
*
dtype0*

seedd*
seed2
ц
Qdefault_policy/sequential_2/twin_q_hidden_0/kernel/Initializer/random_uniform/subSubQdefault_policy/sequential_2/twin_q_hidden_0/kernel/Initializer/random_uniform/maxQdefault_policy/sequential_2/twin_q_hidden_0/kernel/Initializer/random_uniform/min*
T0*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_0/kernel*
_output_shapes
: 
њ
Qdefault_policy/sequential_2/twin_q_hidden_0/kernel/Initializer/random_uniform/mulMul[default_policy/sequential_2/twin_q_hidden_0/kernel/Initializer/random_uniform/RandomUniformQdefault_policy/sequential_2/twin_q_hidden_0/kernel/Initializer/random_uniform/sub*
T0*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_0/kernel* 
_output_shapes
:

ю
Mdefault_policy/sequential_2/twin_q_hidden_0/kernel/Initializer/random_uniformAddV2Qdefault_policy/sequential_2/twin_q_hidden_0/kernel/Initializer/random_uniform/mulQdefault_policy/sequential_2/twin_q_hidden_0/kernel/Initializer/random_uniform/min*
T0*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_0/kernel* 
_output_shapes
:

Б
2default_policy/sequential_2/twin_q_hidden_0/kernelVarHandleOp*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_0/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:
*C
shared_name42default_policy/sequential_2/twin_q_hidden_0/kernel
Е
Sdefault_policy/sequential_2/twin_q_hidden_0/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp2default_policy/sequential_2/twin_q_hidden_0/kernel*
_output_shapes
: 

9default_policy/sequential_2/twin_q_hidden_0/kernel/AssignAssignVariableOp2default_policy/sequential_2/twin_q_hidden_0/kernelMdefault_policy/sequential_2/twin_q_hidden_0/kernel/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Л
Fdefault_policy/sequential_2/twin_q_hidden_0/kernel/Read/ReadVariableOpReadVariableOp2default_policy/sequential_2/twin_q_hidden_0/kernel* 
_output_shapes
:
*
dtype0
ж
Bdefault_policy/sequential_2/twin_q_hidden_0/bias/Initializer/zerosConst*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes	
:*
dtype0*
valueB*    
І
0default_policy/sequential_2/twin_q_hidden_0/biasVarHandleOp*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*A
shared_name20default_policy/sequential_2/twin_q_hidden_0/bias
Б
Qdefault_policy/sequential_2/twin_q_hidden_0/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp0default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes
: 

7default_policy/sequential_2/twin_q_hidden_0/bias/AssignAssignVariableOp0default_policy/sequential_2/twin_q_hidden_0/biasBdefault_policy/sequential_2/twin_q_hidden_0/bias/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
В
Ddefault_policy/sequential_2/twin_q_hidden_0/bias/Read/ReadVariableOpReadVariableOp0default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes	
:*
dtype0
Ж
Adefault_policy/sequential_2/twin_q_hidden_0/MatMul/ReadVariableOpReadVariableOp2default_policy/sequential_2/twin_q_hidden_0/kernel* 
_output_shapes
:
*
dtype0

2default_policy/sequential_2/twin_q_hidden_0/MatMulMatMul0default_policy/sequential_2/concatenate_1/concatAdefault_policy/sequential_2/twin_q_hidden_0/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
А
Bdefault_policy/sequential_2/twin_q_hidden_0/BiasAdd/ReadVariableOpReadVariableOp0default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes	
:*
dtype0

3default_policy/sequential_2/twin_q_hidden_0/BiasAddBiasAdd2default_policy/sequential_2/twin_q_hidden_0/MatMulBdefault_policy/sequential_2/twin_q_hidden_0/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
 
0default_policy/sequential_2/twin_q_hidden_0/ReluRelu3default_policy/sequential_2/twin_q_hidden_0/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
ы
Sdefault_policy/sequential_2/twin_q_hidden_1/kernel/Initializer/random_uniform/shapeConst*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_1/kernel*
_output_shapes
:*
dtype0*
valueB"      
н
Qdefault_policy/sequential_2/twin_q_hidden_1/kernel/Initializer/random_uniform/minConst*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *зГнН
н
Qdefault_policy/sequential_2/twin_q_hidden_1/kernel/Initializer/random_uniform/maxConst*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *зГн=
й
[default_policy/sequential_2/twin_q_hidden_1/kernel/Initializer/random_uniform/RandomUniformRandomUniformSdefault_policy/sequential_2/twin_q_hidden_1/kernel/Initializer/random_uniform/shape*
T0*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_1/kernel* 
_output_shapes
:
*
dtype0*

seedd*
seed2
ц
Qdefault_policy/sequential_2/twin_q_hidden_1/kernel/Initializer/random_uniform/subSubQdefault_policy/sequential_2/twin_q_hidden_1/kernel/Initializer/random_uniform/maxQdefault_policy/sequential_2/twin_q_hidden_1/kernel/Initializer/random_uniform/min*
T0*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_1/kernel*
_output_shapes
: 
њ
Qdefault_policy/sequential_2/twin_q_hidden_1/kernel/Initializer/random_uniform/mulMul[default_policy/sequential_2/twin_q_hidden_1/kernel/Initializer/random_uniform/RandomUniformQdefault_policy/sequential_2/twin_q_hidden_1/kernel/Initializer/random_uniform/sub*
T0*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_1/kernel* 
_output_shapes
:

ю
Mdefault_policy/sequential_2/twin_q_hidden_1/kernel/Initializer/random_uniformAddV2Qdefault_policy/sequential_2/twin_q_hidden_1/kernel/Initializer/random_uniform/mulQdefault_policy/sequential_2/twin_q_hidden_1/kernel/Initializer/random_uniform/min*
T0*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_1/kernel* 
_output_shapes
:

Б
2default_policy/sequential_2/twin_q_hidden_1/kernelVarHandleOp*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_1/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:
*C
shared_name42default_policy/sequential_2/twin_q_hidden_1/kernel
Е
Sdefault_policy/sequential_2/twin_q_hidden_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp2default_policy/sequential_2/twin_q_hidden_1/kernel*
_output_shapes
: 

9default_policy/sequential_2/twin_q_hidden_1/kernel/AssignAssignVariableOp2default_policy/sequential_2/twin_q_hidden_1/kernelMdefault_policy/sequential_2/twin_q_hidden_1/kernel/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Л
Fdefault_policy/sequential_2/twin_q_hidden_1/kernel/Read/ReadVariableOpReadVariableOp2default_policy/sequential_2/twin_q_hidden_1/kernel* 
_output_shapes
:
*
dtype0
ж
Bdefault_policy/sequential_2/twin_q_hidden_1/bias/Initializer/zerosConst*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_1/bias*
_output_shapes	
:*
dtype0*
valueB*    
І
0default_policy/sequential_2/twin_q_hidden_1/biasVarHandleOp*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_1/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*A
shared_name20default_policy/sequential_2/twin_q_hidden_1/bias
Б
Qdefault_policy/sequential_2/twin_q_hidden_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp0default_policy/sequential_2/twin_q_hidden_1/bias*
_output_shapes
: 

7default_policy/sequential_2/twin_q_hidden_1/bias/AssignAssignVariableOp0default_policy/sequential_2/twin_q_hidden_1/biasBdefault_policy/sequential_2/twin_q_hidden_1/bias/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
В
Ddefault_policy/sequential_2/twin_q_hidden_1/bias/Read/ReadVariableOpReadVariableOp0default_policy/sequential_2/twin_q_hidden_1/bias*
_output_shapes	
:*
dtype0
Ж
Adefault_policy/sequential_2/twin_q_hidden_1/MatMul/ReadVariableOpReadVariableOp2default_policy/sequential_2/twin_q_hidden_1/kernel* 
_output_shapes
:
*
dtype0

2default_policy/sequential_2/twin_q_hidden_1/MatMulMatMul0default_policy/sequential_2/twin_q_hidden_0/ReluAdefault_policy/sequential_2/twin_q_hidden_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
А
Bdefault_policy/sequential_2/twin_q_hidden_1/BiasAdd/ReadVariableOpReadVariableOp0default_policy/sequential_2/twin_q_hidden_1/bias*
_output_shapes	
:*
dtype0

3default_policy/sequential_2/twin_q_hidden_1/BiasAddBiasAdd2default_policy/sequential_2/twin_q_hidden_1/MatMulBdefault_policy/sequential_2/twin_q_hidden_1/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
 
0default_policy/sequential_2/twin_q_hidden_1/ReluRelu3default_policy/sequential_2/twin_q_hidden_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
с
Ndefault_policy/sequential_2/twin_q_out/kernel/Initializer/random_uniform/shapeConst*@
_class6
42loc:@default_policy/sequential_2/twin_q_out/kernel*
_output_shapes
:*
dtype0*
valueB"      
г
Ldefault_policy/sequential_2/twin_q_out/kernel/Initializer/random_uniform/minConst*@
_class6
42loc:@default_policy/sequential_2/twin_q_out/kernel*
_output_shapes
: *
dtype0*
valueB
 *IvО
г
Ldefault_policy/sequential_2/twin_q_out/kernel/Initializer/random_uniform/maxConst*@
_class6
42loc:@default_policy/sequential_2/twin_q_out/kernel*
_output_shapes
: *
dtype0*
valueB
 *Iv>
Щ
Vdefault_policy/sequential_2/twin_q_out/kernel/Initializer/random_uniform/RandomUniformRandomUniformNdefault_policy/sequential_2/twin_q_out/kernel/Initializer/random_uniform/shape*
T0*@
_class6
42loc:@default_policy/sequential_2/twin_q_out/kernel*
_output_shapes
:	*
dtype0*

seedd*
seed2
в
Ldefault_policy/sequential_2/twin_q_out/kernel/Initializer/random_uniform/subSubLdefault_policy/sequential_2/twin_q_out/kernel/Initializer/random_uniform/maxLdefault_policy/sequential_2/twin_q_out/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@default_policy/sequential_2/twin_q_out/kernel*
_output_shapes
: 
х
Ldefault_policy/sequential_2/twin_q_out/kernel/Initializer/random_uniform/mulMulVdefault_policy/sequential_2/twin_q_out/kernel/Initializer/random_uniform/RandomUniformLdefault_policy/sequential_2/twin_q_out/kernel/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@default_policy/sequential_2/twin_q_out/kernel*
_output_shapes
:	
й
Hdefault_policy/sequential_2/twin_q_out/kernel/Initializer/random_uniformAddV2Ldefault_policy/sequential_2/twin_q_out/kernel/Initializer/random_uniform/mulLdefault_policy/sequential_2/twin_q_out/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@default_policy/sequential_2/twin_q_out/kernel*
_output_shapes
:	
Ё
-default_policy/sequential_2/twin_q_out/kernelVarHandleOp*@
_class6
42loc:@default_policy/sequential_2/twin_q_out/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:	*>
shared_name/-default_policy/sequential_2/twin_q_out/kernel
Ћ
Ndefault_policy/sequential_2/twin_q_out/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp-default_policy/sequential_2/twin_q_out/kernel*
_output_shapes
: 

4default_policy/sequential_2/twin_q_out/kernel/AssignAssignVariableOp-default_policy/sequential_2/twin_q_out/kernelHdefault_policy/sequential_2/twin_q_out/kernel/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
А
Adefault_policy/sequential_2/twin_q_out/kernel/Read/ReadVariableOpReadVariableOp-default_policy/sequential_2/twin_q_out/kernel*
_output_shapes
:	*
dtype0
Ъ
=default_policy/sequential_2/twin_q_out/bias/Initializer/zerosConst*>
_class4
20loc:@default_policy/sequential_2/twin_q_out/bias*
_output_shapes
:*
dtype0*
valueB*    

+default_policy/sequential_2/twin_q_out/biasVarHandleOp*>
_class4
20loc:@default_policy/sequential_2/twin_q_out/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*<
shared_name-+default_policy/sequential_2/twin_q_out/bias
Ї
Ldefault_policy/sequential_2/twin_q_out/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp+default_policy/sequential_2/twin_q_out/bias*
_output_shapes
: 
§
2default_policy/sequential_2/twin_q_out/bias/AssignAssignVariableOp+default_policy/sequential_2/twin_q_out/bias=default_policy/sequential_2/twin_q_out/bias/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Ї
?default_policy/sequential_2/twin_q_out/bias/Read/ReadVariableOpReadVariableOp+default_policy/sequential_2/twin_q_out/bias*
_output_shapes
:*
dtype0
Ћ
<default_policy/sequential_2/twin_q_out/MatMul/ReadVariableOpReadVariableOp-default_policy/sequential_2/twin_q_out/kernel*
_output_shapes
:	*
dtype0
џ
-default_policy/sequential_2/twin_q_out/MatMulMatMul0default_policy/sequential_2/twin_q_hidden_1/Relu<default_policy/sequential_2/twin_q_out/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ѕ
=default_policy/sequential_2/twin_q_out/BiasAdd/ReadVariableOpReadVariableOp+default_policy/sequential_2/twin_q_out/bias*
_output_shapes
:*
dtype0
№
.default_policy/sequential_2/twin_q_out/BiasAddBiasAdd-default_policy/sequential_2/twin_q_out/MatMul=default_policy/sequential_2/twin_q_out/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
Є
2default_policy/log_alpha/Initializer/initial_valueConst*+
_class!
loc:@default_policy/log_alpha*
_output_shapes
: *
dtype0*
valueB
 *    
й
default_policy/log_alphaVarHandleOp*+
_class!
loc:@default_policy/log_alpha*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *)
shared_namedefault_policy/log_alpha

9default_policy/log_alpha/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/log_alpha*
_output_shapes
: 
Ь
default_policy/log_alpha/AssignAssignVariableOpdefault_policy/log_alpha2default_policy/log_alpha/Initializer/initial_value*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
}
,default_policy/log_alpha/Read/ReadVariableOpReadVariableOpdefault_policy/log_alpha*
_output_shapes
: *
dtype0
r
!default_policy/Exp/ReadVariableOpReadVariableOpdefault_policy/log_alpha*
_output_shapes
: *
dtype0
]
default_policy/ExpExp!default_policy/Exp/ReadVariableOp*
T0*
_output_shapes
: 

default_policy/observations_1Placeholder*(
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
Ь	
3default_policy/value_out_1/kernel/Initializer/ConstConst*4
_class*
(&loc:@default_policy/value_out_1/kernel*
_output_shapes
:	*
dtype0*Љ
valueB	"?П:Пs9б&КїЏ@:1lЙ~?КъЖП94ЙЄИrЙфЁЙаPј931s7Јцd:)_9ўЙў8:_Р%:wм КYх9I,Йэ9/9*OЬ:зF:ћxс9jцБЙКcѕЙvеИі9кѕ ЙE Къ3k:vб$КЎф9ы4Кoгд8ФКќъ8
К пКZм8UwЅЙоўЗ1Uя:vћ:'йq:QШ::мЯJЙзu9"КлЎ9бP8g9ПSI:+|:Yк<Й:8ti:0ИY:мф:sђU8ЕЫGКCд9K4:-;Йоs+Кгэ9S9q`ИтЙ<rАЙD*КZbp9|LКгКqaП8Iд :@=Э:lЪК
йЂ9мЃ9\^:Я=:ЩаЙї4ЙыЩС6
ф9КhJЊЙu/КlLЗuРG9
 ]ЙБЪ:ХШЙєг`:№јpЙ6D9?aХЙe8ЗЪХ:IbИ$ЮИИsю9СЏ$:htЙмдЁ9ЈН К"К9Й?К$:ЊКЭУi7ц7КFу8/ЃЌЙuЯ:Ш8Ц­ К
эЕ9п КіЛdКЌF9КЏЩ8§озИэИ1:аУЙoьо9јкКИЖЏOИJЙгоЄИш_8ююф8КУrь9.ЙDЇ2К~ЎюИ ъ КщЬc7АЭИuX!КuB9рКPs|9м­К@ЌУ8#ЎЙsшpЙn`:rуWЙ`W9H:џЇe:РRКў4*:нКъ3КJaO:&дЙW9[З:хs:МSИ.ЃЙ]oЙЏьЁЗsЭ:жКZР'ЙІ§ц9А}р9ќѕКьk:8!ъКvA КFуНЙb6Йја:ЧlКЮЖ@КKpК'Z@Й`1К>?:^
7шХ9epЈ95Є8gKЙОБIКЮExЙЇИ8Р99:n5:@Кx_5ИнT9Бь7ЗњA:ъ:!ЙОG:s!ЙЙЄWКџђ9YL:йљ9ЙАф9.дЙDЖ6:T№К=.^ЙЖ~iКBЙs+КTіЃИB`П9эсЗ	Ьb9уНЙцА:]њ$:ЈГPКЇ]:PЁ:З:И69ѓЭCКfЧџ9№9"Й369XЋКжф9#$:Ђ:gbB:8џЋИаG{:4,PКѓz;Ёєч:Ѓњ9ПyКХЈ8qєж9њX9з9t6!:ЂaЙr 8'HИ
§
!default_policy/value_out_1/kernelVarHandleOp*4
_class*
(&loc:@default_policy/value_out_1/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:	*2
shared_name#!default_policy/value_out_1/kernel

Bdefault_policy/value_out_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp!default_policy/value_out_1/kernel*
_output_shapes
: 
п
(default_policy/value_out_1/kernel/AssignAssignVariableOp!default_policy/value_out_1/kernel3default_policy/value_out_1/kernel/Initializer/Const*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

5default_policy/value_out_1/kernel/Read/ReadVariableOpReadVariableOp!default_policy/value_out_1/kernel*
_output_shapes
:	*
dtype0
В
1default_policy/value_out_1/bias/Initializer/zerosConst*2
_class(
&$loc:@default_policy/value_out_1/bias*
_output_shapes
:*
dtype0*
valueB*    
ђ
default_policy/value_out_1/biasVarHandleOp*2
_class(
&$loc:@default_policy/value_out_1/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*0
shared_name!default_policy/value_out_1/bias

@default_policy/value_out_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/value_out_1/bias*
_output_shapes
: 
й
&default_policy/value_out_1/bias/AssignAssignVariableOpdefault_policy/value_out_1/bias1default_policy/value_out_1/bias/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

3default_policy/value_out_1/bias/Read/ReadVariableOpReadVariableOpdefault_policy/value_out_1/bias*
_output_shapes
:*
dtype0

0default_policy/value_out_1/MatMul/ReadVariableOpReadVariableOp!default_policy/value_out_1/kernel*
_output_shapes
:	*
dtype0
д
!default_policy/value_out_1/MatMulMatMuldefault_policy/observations_10default_policy/value_out_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

1default_policy/value_out_1/BiasAdd/ReadVariableOpReadVariableOpdefault_policy/value_out_1/bias*
_output_shapes
:*
dtype0
Ь
"default_policy/value_out_1/BiasAddBiasAdd!default_policy/value_out_1/MatMul1default_policy/value_out_1/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC

default_policy/model_out_1Placeholder*(
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
н
Ldefault_policy/sequential_3/action_1/kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@default_policy/sequential_3/action_1/kernel*
_output_shapes
:*
dtype0*
valueB"     
Я
Jdefault_policy/sequential_3/action_1/kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@default_policy/sequential_3/action_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *KнН
Я
Jdefault_policy/sequential_3/action_1/kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@default_policy/sequential_3/action_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *Kн=
Ф
Tdefault_policy/sequential_3/action_1/kernel/Initializer/random_uniform/RandomUniformRandomUniformLdefault_policy/sequential_3/action_1/kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@default_policy/sequential_3/action_1/kernel* 
_output_shapes
:
*
dtype0*

seedd*
seed2	
Ъ
Jdefault_policy/sequential_3/action_1/kernel/Initializer/random_uniform/subSubJdefault_policy/sequential_3/action_1/kernel/Initializer/random_uniform/maxJdefault_policy/sequential_3/action_1/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@default_policy/sequential_3/action_1/kernel*
_output_shapes
: 
о
Jdefault_policy/sequential_3/action_1/kernel/Initializer/random_uniform/mulMulTdefault_policy/sequential_3/action_1/kernel/Initializer/random_uniform/RandomUniformJdefault_policy/sequential_3/action_1/kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@default_policy/sequential_3/action_1/kernel* 
_output_shapes
:

в
Fdefault_policy/sequential_3/action_1/kernel/Initializer/random_uniformAddV2Jdefault_policy/sequential_3/action_1/kernel/Initializer/random_uniform/mulJdefault_policy/sequential_3/action_1/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@default_policy/sequential_3/action_1/kernel* 
_output_shapes
:


+default_policy/sequential_3/action_1/kernelVarHandleOp*>
_class4
20loc:@default_policy/sequential_3/action_1/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:
*<
shared_name-+default_policy/sequential_3/action_1/kernel
Ї
Ldefault_policy/sequential_3/action_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+default_policy/sequential_3/action_1/kernel*
_output_shapes
: 

2default_policy/sequential_3/action_1/kernel/AssignAssignVariableOp+default_policy/sequential_3/action_1/kernelFdefault_policy/sequential_3/action_1/kernel/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
­
?default_policy/sequential_3/action_1/kernel/Read/ReadVariableOpReadVariableOp+default_policy/sequential_3/action_1/kernel* 
_output_shapes
:
*
dtype0
Ш
;default_policy/sequential_3/action_1/bias/Initializer/zerosConst*<
_class2
0.loc:@default_policy/sequential_3/action_1/bias*
_output_shapes	
:*
dtype0*
valueB*    

)default_policy/sequential_3/action_1/biasVarHandleOp*<
_class2
0.loc:@default_policy/sequential_3/action_1/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*:
shared_name+)default_policy/sequential_3/action_1/bias
Ѓ
Jdefault_policy/sequential_3/action_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp)default_policy/sequential_3/action_1/bias*
_output_shapes
: 
ї
0default_policy/sequential_3/action_1/bias/AssignAssignVariableOp)default_policy/sequential_3/action_1/bias;default_policy/sequential_3/action_1/bias/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Є
=default_policy/sequential_3/action_1/bias/Read/ReadVariableOpReadVariableOp)default_policy/sequential_3/action_1/bias*
_output_shapes	
:*
dtype0
Ј
:default_policy/sequential_3/action_1/MatMul/ReadVariableOpReadVariableOp+default_policy/sequential_3/action_1/kernel* 
_output_shapes
:
*
dtype0
ц
+default_policy/sequential_3/action_1/MatMulMatMuldefault_policy/model_out_1:default_policy/sequential_3/action_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ђ
;default_policy/sequential_3/action_1/BiasAdd/ReadVariableOpReadVariableOp)default_policy/sequential_3/action_1/bias*
_output_shapes	
:*
dtype0
ы
,default_policy/sequential_3/action_1/BiasAddBiasAdd+default_policy/sequential_3/action_1/MatMul;default_policy/sequential_3/action_1/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC

)default_policy/sequential_3/action_1/ReluRelu,default_policy/sequential_3/action_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
н
Ldefault_policy/sequential_3/action_2/kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@default_policy/sequential_3/action_2/kernel*
_output_shapes
:*
dtype0*
valueB"      
Я
Jdefault_policy/sequential_3/action_2/kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@default_policy/sequential_3/action_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *зГнН
Я
Jdefault_policy/sequential_3/action_2/kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@default_policy/sequential_3/action_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *зГн=
Ф
Tdefault_policy/sequential_3/action_2/kernel/Initializer/random_uniform/RandomUniformRandomUniformLdefault_policy/sequential_3/action_2/kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@default_policy/sequential_3/action_2/kernel* 
_output_shapes
:
*
dtype0*

seedd*
seed2

Ъ
Jdefault_policy/sequential_3/action_2/kernel/Initializer/random_uniform/subSubJdefault_policy/sequential_3/action_2/kernel/Initializer/random_uniform/maxJdefault_policy/sequential_3/action_2/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@default_policy/sequential_3/action_2/kernel*
_output_shapes
: 
о
Jdefault_policy/sequential_3/action_2/kernel/Initializer/random_uniform/mulMulTdefault_policy/sequential_3/action_2/kernel/Initializer/random_uniform/RandomUniformJdefault_policy/sequential_3/action_2/kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@default_policy/sequential_3/action_2/kernel* 
_output_shapes
:

в
Fdefault_policy/sequential_3/action_2/kernel/Initializer/random_uniformAddV2Jdefault_policy/sequential_3/action_2/kernel/Initializer/random_uniform/mulJdefault_policy/sequential_3/action_2/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@default_policy/sequential_3/action_2/kernel* 
_output_shapes
:


+default_policy/sequential_3/action_2/kernelVarHandleOp*>
_class4
20loc:@default_policy/sequential_3/action_2/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:
*<
shared_name-+default_policy/sequential_3/action_2/kernel
Ї
Ldefault_policy/sequential_3/action_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp+default_policy/sequential_3/action_2/kernel*
_output_shapes
: 

2default_policy/sequential_3/action_2/kernel/AssignAssignVariableOp+default_policy/sequential_3/action_2/kernelFdefault_policy/sequential_3/action_2/kernel/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
­
?default_policy/sequential_3/action_2/kernel/Read/ReadVariableOpReadVariableOp+default_policy/sequential_3/action_2/kernel* 
_output_shapes
:
*
dtype0
Ш
;default_policy/sequential_3/action_2/bias/Initializer/zerosConst*<
_class2
0.loc:@default_policy/sequential_3/action_2/bias*
_output_shapes	
:*
dtype0*
valueB*    

)default_policy/sequential_3/action_2/biasVarHandleOp*<
_class2
0.loc:@default_policy/sequential_3/action_2/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*:
shared_name+)default_policy/sequential_3/action_2/bias
Ѓ
Jdefault_policy/sequential_3/action_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp)default_policy/sequential_3/action_2/bias*
_output_shapes
: 
ї
0default_policy/sequential_3/action_2/bias/AssignAssignVariableOp)default_policy/sequential_3/action_2/bias;default_policy/sequential_3/action_2/bias/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Є
=default_policy/sequential_3/action_2/bias/Read/ReadVariableOpReadVariableOp)default_policy/sequential_3/action_2/bias*
_output_shapes	
:*
dtype0
Ј
:default_policy/sequential_3/action_2/MatMul/ReadVariableOpReadVariableOp+default_policy/sequential_3/action_2/kernel* 
_output_shapes
:
*
dtype0
ѕ
+default_policy/sequential_3/action_2/MatMulMatMul)default_policy/sequential_3/action_1/Relu:default_policy/sequential_3/action_2/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ђ
;default_policy/sequential_3/action_2/BiasAdd/ReadVariableOpReadVariableOp)default_policy/sequential_3/action_2/bias*
_output_shapes	
:*
dtype0
ы
,default_policy/sequential_3/action_2/BiasAddBiasAdd+default_policy/sequential_3/action_2/MatMul;default_policy/sequential_3/action_2/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC

)default_policy/sequential_3/action_2/ReluRelu,default_policy/sequential_3/action_2/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
с
Ndefault_policy/sequential_3/action_out/kernel/Initializer/random_uniform/shapeConst*@
_class6
42loc:@default_policy/sequential_3/action_out/kernel*
_output_shapes
:*
dtype0*
valueB"      
г
Ldefault_policy/sequential_3/action_out/kernel/Initializer/random_uniform/minConst*@
_class6
42loc:@default_policy/sequential_3/action_out/kernel*
_output_shapes
: *
dtype0*
valueB
 *О
г
Ldefault_policy/sequential_3/action_out/kernel/Initializer/random_uniform/maxConst*@
_class6
42loc:@default_policy/sequential_3/action_out/kernel*
_output_shapes
: *
dtype0*
valueB
 *>
Щ
Vdefault_policy/sequential_3/action_out/kernel/Initializer/random_uniform/RandomUniformRandomUniformNdefault_policy/sequential_3/action_out/kernel/Initializer/random_uniform/shape*
T0*@
_class6
42loc:@default_policy/sequential_3/action_out/kernel*
_output_shapes
:	*
dtype0*

seedd*
seed2
в
Ldefault_policy/sequential_3/action_out/kernel/Initializer/random_uniform/subSubLdefault_policy/sequential_3/action_out/kernel/Initializer/random_uniform/maxLdefault_policy/sequential_3/action_out/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@default_policy/sequential_3/action_out/kernel*
_output_shapes
: 
х
Ldefault_policy/sequential_3/action_out/kernel/Initializer/random_uniform/mulMulVdefault_policy/sequential_3/action_out/kernel/Initializer/random_uniform/RandomUniformLdefault_policy/sequential_3/action_out/kernel/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@default_policy/sequential_3/action_out/kernel*
_output_shapes
:	
й
Hdefault_policy/sequential_3/action_out/kernel/Initializer/random_uniformAddV2Ldefault_policy/sequential_3/action_out/kernel/Initializer/random_uniform/mulLdefault_policy/sequential_3/action_out/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@default_policy/sequential_3/action_out/kernel*
_output_shapes
:	
Ё
-default_policy/sequential_3/action_out/kernelVarHandleOp*@
_class6
42loc:@default_policy/sequential_3/action_out/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:	*>
shared_name/-default_policy/sequential_3/action_out/kernel
Ћ
Ndefault_policy/sequential_3/action_out/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp-default_policy/sequential_3/action_out/kernel*
_output_shapes
: 

4default_policy/sequential_3/action_out/kernel/AssignAssignVariableOp-default_policy/sequential_3/action_out/kernelHdefault_policy/sequential_3/action_out/kernel/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
А
Adefault_policy/sequential_3/action_out/kernel/Read/ReadVariableOpReadVariableOp-default_policy/sequential_3/action_out/kernel*
_output_shapes
:	*
dtype0
Ъ
=default_policy/sequential_3/action_out/bias/Initializer/zerosConst*>
_class4
20loc:@default_policy/sequential_3/action_out/bias*
_output_shapes
:*
dtype0*
valueB*    

+default_policy/sequential_3/action_out/biasVarHandleOp*>
_class4
20loc:@default_policy/sequential_3/action_out/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*<
shared_name-+default_policy/sequential_3/action_out/bias
Ї
Ldefault_policy/sequential_3/action_out/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp+default_policy/sequential_3/action_out/bias*
_output_shapes
: 
§
2default_policy/sequential_3/action_out/bias/AssignAssignVariableOp+default_policy/sequential_3/action_out/bias=default_policy/sequential_3/action_out/bias/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Ї
?default_policy/sequential_3/action_out/bias/Read/ReadVariableOpReadVariableOp+default_policy/sequential_3/action_out/bias*
_output_shapes
:*
dtype0
Ћ
<default_policy/sequential_3/action_out/MatMul/ReadVariableOpReadVariableOp-default_policy/sequential_3/action_out/kernel*
_output_shapes
:	*
dtype0
ј
-default_policy/sequential_3/action_out/MatMulMatMul)default_policy/sequential_3/action_2/Relu<default_policy/sequential_3/action_out/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ѕ
=default_policy/sequential_3/action_out/BiasAdd/ReadVariableOpReadVariableOp+default_policy/sequential_3/action_out/bias*
_output_shapes
:*
dtype0
№
.default_policy/sequential_3/action_out/BiasAddBiasAdd-default_policy/sequential_3/action_out/MatMul=default_policy/sequential_3/action_out/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
{
default_policy/actions_1Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
w
5default_policy/sequential_4/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
ё
0default_policy/sequential_4/concatenate_2/concatConcatV2default_policy/model_out_1default_policy/actions_15default_policy/sequential_4/concatenate_2/concat/axis*
N*
T0*

Tidx0*(
_output_shapes
:џџџџџџџџџ
с
Ndefault_policy/sequential_4/q_hidden_0/kernel/Initializer/random_uniform/shapeConst*@
_class6
42loc:@default_policy/sequential_4/q_hidden_0/kernel*
_output_shapes
:*
dtype0*
valueB"     
г
Ldefault_policy/sequential_4/q_hidden_0/kernel/Initializer/random_uniform/minConst*@
_class6
42loc:@default_policy/sequential_4/q_hidden_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *К мН
г
Ldefault_policy/sequential_4/q_hidden_0/kernel/Initializer/random_uniform/maxConst*@
_class6
42loc:@default_policy/sequential_4/q_hidden_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *К м=
Ъ
Vdefault_policy/sequential_4/q_hidden_0/kernel/Initializer/random_uniform/RandomUniformRandomUniformNdefault_policy/sequential_4/q_hidden_0/kernel/Initializer/random_uniform/shape*
T0*@
_class6
42loc:@default_policy/sequential_4/q_hidden_0/kernel* 
_output_shapes
:
*
dtype0*

seedd*
seed2
в
Ldefault_policy/sequential_4/q_hidden_0/kernel/Initializer/random_uniform/subSubLdefault_policy/sequential_4/q_hidden_0/kernel/Initializer/random_uniform/maxLdefault_policy/sequential_4/q_hidden_0/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@default_policy/sequential_4/q_hidden_0/kernel*
_output_shapes
: 
ц
Ldefault_policy/sequential_4/q_hidden_0/kernel/Initializer/random_uniform/mulMulVdefault_policy/sequential_4/q_hidden_0/kernel/Initializer/random_uniform/RandomUniformLdefault_policy/sequential_4/q_hidden_0/kernel/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@default_policy/sequential_4/q_hidden_0/kernel* 
_output_shapes
:

к
Hdefault_policy/sequential_4/q_hidden_0/kernel/Initializer/random_uniformAddV2Ldefault_policy/sequential_4/q_hidden_0/kernel/Initializer/random_uniform/mulLdefault_policy/sequential_4/q_hidden_0/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@default_policy/sequential_4/q_hidden_0/kernel* 
_output_shapes
:

Ђ
-default_policy/sequential_4/q_hidden_0/kernelVarHandleOp*@
_class6
42loc:@default_policy/sequential_4/q_hidden_0/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:
*>
shared_name/-default_policy/sequential_4/q_hidden_0/kernel
Ћ
Ndefault_policy/sequential_4/q_hidden_0/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp-default_policy/sequential_4/q_hidden_0/kernel*
_output_shapes
: 

4default_policy/sequential_4/q_hidden_0/kernel/AssignAssignVariableOp-default_policy/sequential_4/q_hidden_0/kernelHdefault_policy/sequential_4/q_hidden_0/kernel/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Б
Adefault_policy/sequential_4/q_hidden_0/kernel/Read/ReadVariableOpReadVariableOp-default_policy/sequential_4/q_hidden_0/kernel* 
_output_shapes
:
*
dtype0
Ь
=default_policy/sequential_4/q_hidden_0/bias/Initializer/zerosConst*>
_class4
20loc:@default_policy/sequential_4/q_hidden_0/bias*
_output_shapes	
:*
dtype0*
valueB*    

+default_policy/sequential_4/q_hidden_0/biasVarHandleOp*>
_class4
20loc:@default_policy/sequential_4/q_hidden_0/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*<
shared_name-+default_policy/sequential_4/q_hidden_0/bias
Ї
Ldefault_policy/sequential_4/q_hidden_0/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp+default_policy/sequential_4/q_hidden_0/bias*
_output_shapes
: 
§
2default_policy/sequential_4/q_hidden_0/bias/AssignAssignVariableOp+default_policy/sequential_4/q_hidden_0/bias=default_policy/sequential_4/q_hidden_0/bias/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Ј
?default_policy/sequential_4/q_hidden_0/bias/Read/ReadVariableOpReadVariableOp+default_policy/sequential_4/q_hidden_0/bias*
_output_shapes	
:*
dtype0
Ќ
<default_policy/sequential_4/q_hidden_0/MatMul/ReadVariableOpReadVariableOp-default_policy/sequential_4/q_hidden_0/kernel* 
_output_shapes
:
*
dtype0

-default_policy/sequential_4/q_hidden_0/MatMulMatMul0default_policy/sequential_4/concatenate_2/concat<default_policy/sequential_4/q_hidden_0/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
І
=default_policy/sequential_4/q_hidden_0/BiasAdd/ReadVariableOpReadVariableOp+default_policy/sequential_4/q_hidden_0/bias*
_output_shapes	
:*
dtype0
ё
.default_policy/sequential_4/q_hidden_0/BiasAddBiasAdd-default_policy/sequential_4/q_hidden_0/MatMul=default_policy/sequential_4/q_hidden_0/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC

+default_policy/sequential_4/q_hidden_0/ReluRelu.default_policy/sequential_4/q_hidden_0/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
с
Ndefault_policy/sequential_4/q_hidden_1/kernel/Initializer/random_uniform/shapeConst*@
_class6
42loc:@default_policy/sequential_4/q_hidden_1/kernel*
_output_shapes
:*
dtype0*
valueB"      
г
Ldefault_policy/sequential_4/q_hidden_1/kernel/Initializer/random_uniform/minConst*@
_class6
42loc:@default_policy/sequential_4/q_hidden_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *зГнН
г
Ldefault_policy/sequential_4/q_hidden_1/kernel/Initializer/random_uniform/maxConst*@
_class6
42loc:@default_policy/sequential_4/q_hidden_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *зГн=
Ъ
Vdefault_policy/sequential_4/q_hidden_1/kernel/Initializer/random_uniform/RandomUniformRandomUniformNdefault_policy/sequential_4/q_hidden_1/kernel/Initializer/random_uniform/shape*
T0*@
_class6
42loc:@default_policy/sequential_4/q_hidden_1/kernel* 
_output_shapes
:
*
dtype0*

seedd*
seed2
в
Ldefault_policy/sequential_4/q_hidden_1/kernel/Initializer/random_uniform/subSubLdefault_policy/sequential_4/q_hidden_1/kernel/Initializer/random_uniform/maxLdefault_policy/sequential_4/q_hidden_1/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@default_policy/sequential_4/q_hidden_1/kernel*
_output_shapes
: 
ц
Ldefault_policy/sequential_4/q_hidden_1/kernel/Initializer/random_uniform/mulMulVdefault_policy/sequential_4/q_hidden_1/kernel/Initializer/random_uniform/RandomUniformLdefault_policy/sequential_4/q_hidden_1/kernel/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@default_policy/sequential_4/q_hidden_1/kernel* 
_output_shapes
:

к
Hdefault_policy/sequential_4/q_hidden_1/kernel/Initializer/random_uniformAddV2Ldefault_policy/sequential_4/q_hidden_1/kernel/Initializer/random_uniform/mulLdefault_policy/sequential_4/q_hidden_1/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@default_policy/sequential_4/q_hidden_1/kernel* 
_output_shapes
:

Ђ
-default_policy/sequential_4/q_hidden_1/kernelVarHandleOp*@
_class6
42loc:@default_policy/sequential_4/q_hidden_1/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:
*>
shared_name/-default_policy/sequential_4/q_hidden_1/kernel
Ћ
Ndefault_policy/sequential_4/q_hidden_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp-default_policy/sequential_4/q_hidden_1/kernel*
_output_shapes
: 

4default_policy/sequential_4/q_hidden_1/kernel/AssignAssignVariableOp-default_policy/sequential_4/q_hidden_1/kernelHdefault_policy/sequential_4/q_hidden_1/kernel/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Б
Adefault_policy/sequential_4/q_hidden_1/kernel/Read/ReadVariableOpReadVariableOp-default_policy/sequential_4/q_hidden_1/kernel* 
_output_shapes
:
*
dtype0
Ь
=default_policy/sequential_4/q_hidden_1/bias/Initializer/zerosConst*>
_class4
20loc:@default_policy/sequential_4/q_hidden_1/bias*
_output_shapes	
:*
dtype0*
valueB*    

+default_policy/sequential_4/q_hidden_1/biasVarHandleOp*>
_class4
20loc:@default_policy/sequential_4/q_hidden_1/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*<
shared_name-+default_policy/sequential_4/q_hidden_1/bias
Ї
Ldefault_policy/sequential_4/q_hidden_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp+default_policy/sequential_4/q_hidden_1/bias*
_output_shapes
: 
§
2default_policy/sequential_4/q_hidden_1/bias/AssignAssignVariableOp+default_policy/sequential_4/q_hidden_1/bias=default_policy/sequential_4/q_hidden_1/bias/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Ј
?default_policy/sequential_4/q_hidden_1/bias/Read/ReadVariableOpReadVariableOp+default_policy/sequential_4/q_hidden_1/bias*
_output_shapes	
:*
dtype0
Ќ
<default_policy/sequential_4/q_hidden_1/MatMul/ReadVariableOpReadVariableOp-default_policy/sequential_4/q_hidden_1/kernel* 
_output_shapes
:
*
dtype0
ћ
-default_policy/sequential_4/q_hidden_1/MatMulMatMul+default_policy/sequential_4/q_hidden_0/Relu<default_policy/sequential_4/q_hidden_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
І
=default_policy/sequential_4/q_hidden_1/BiasAdd/ReadVariableOpReadVariableOp+default_policy/sequential_4/q_hidden_1/bias*
_output_shapes	
:*
dtype0
ё
.default_policy/sequential_4/q_hidden_1/BiasAddBiasAdd-default_policy/sequential_4/q_hidden_1/MatMul=default_policy/sequential_4/q_hidden_1/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC

+default_policy/sequential_4/q_hidden_1/ReluRelu.default_policy/sequential_4/q_hidden_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
з
Idefault_policy/sequential_4/q_out/kernel/Initializer/random_uniform/shapeConst*;
_class1
/-loc:@default_policy/sequential_4/q_out/kernel*
_output_shapes
:*
dtype0*
valueB"      
Щ
Gdefault_policy/sequential_4/q_out/kernel/Initializer/random_uniform/minConst*;
_class1
/-loc:@default_policy/sequential_4/q_out/kernel*
_output_shapes
: *
dtype0*
valueB
 *IvО
Щ
Gdefault_policy/sequential_4/q_out/kernel/Initializer/random_uniform/maxConst*;
_class1
/-loc:@default_policy/sequential_4/q_out/kernel*
_output_shapes
: *
dtype0*
valueB
 *Iv>
К
Qdefault_policy/sequential_4/q_out/kernel/Initializer/random_uniform/RandomUniformRandomUniformIdefault_policy/sequential_4/q_out/kernel/Initializer/random_uniform/shape*
T0*;
_class1
/-loc:@default_policy/sequential_4/q_out/kernel*
_output_shapes
:	*
dtype0*

seedd*
seed2
О
Gdefault_policy/sequential_4/q_out/kernel/Initializer/random_uniform/subSubGdefault_policy/sequential_4/q_out/kernel/Initializer/random_uniform/maxGdefault_policy/sequential_4/q_out/kernel/Initializer/random_uniform/min*
T0*;
_class1
/-loc:@default_policy/sequential_4/q_out/kernel*
_output_shapes
: 
б
Gdefault_policy/sequential_4/q_out/kernel/Initializer/random_uniform/mulMulQdefault_policy/sequential_4/q_out/kernel/Initializer/random_uniform/RandomUniformGdefault_policy/sequential_4/q_out/kernel/Initializer/random_uniform/sub*
T0*;
_class1
/-loc:@default_policy/sequential_4/q_out/kernel*
_output_shapes
:	
Х
Cdefault_policy/sequential_4/q_out/kernel/Initializer/random_uniformAddV2Gdefault_policy/sequential_4/q_out/kernel/Initializer/random_uniform/mulGdefault_policy/sequential_4/q_out/kernel/Initializer/random_uniform/min*
T0*;
_class1
/-loc:@default_policy/sequential_4/q_out/kernel*
_output_shapes
:	

(default_policy/sequential_4/q_out/kernelVarHandleOp*;
_class1
/-loc:@default_policy/sequential_4/q_out/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:	*9
shared_name*(default_policy/sequential_4/q_out/kernel
Ё
Idefault_policy/sequential_4/q_out/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp(default_policy/sequential_4/q_out/kernel*
_output_shapes
: 
§
/default_policy/sequential_4/q_out/kernel/AssignAssignVariableOp(default_policy/sequential_4/q_out/kernelCdefault_policy/sequential_4/q_out/kernel/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
І
<default_policy/sequential_4/q_out/kernel/Read/ReadVariableOpReadVariableOp(default_policy/sequential_4/q_out/kernel*
_output_shapes
:	*
dtype0
Р
8default_policy/sequential_4/q_out/bias/Initializer/zerosConst*9
_class/
-+loc:@default_policy/sequential_4/q_out/bias*
_output_shapes
:*
dtype0*
valueB*    

&default_policy/sequential_4/q_out/biasVarHandleOp*9
_class/
-+loc:@default_policy/sequential_4/q_out/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*7
shared_name(&default_policy/sequential_4/q_out/bias

Gdefault_policy/sequential_4/q_out/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp&default_policy/sequential_4/q_out/bias*
_output_shapes
: 
ю
-default_policy/sequential_4/q_out/bias/AssignAssignVariableOp&default_policy/sequential_4/q_out/bias8default_policy/sequential_4/q_out/bias/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

:default_policy/sequential_4/q_out/bias/Read/ReadVariableOpReadVariableOp&default_policy/sequential_4/q_out/bias*
_output_shapes
:*
dtype0
Ё
7default_policy/sequential_4/q_out/MatMul/ReadVariableOpReadVariableOp(default_policy/sequential_4/q_out/kernel*
_output_shapes
:	*
dtype0
№
(default_policy/sequential_4/q_out/MatMulMatMul+default_policy/sequential_4/q_hidden_1/Relu7default_policy/sequential_4/q_out/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

8default_policy/sequential_4/q_out/BiasAdd/ReadVariableOpReadVariableOp&default_policy/sequential_4/q_out/bias*
_output_shapes
:*
dtype0
с
)default_policy/sequential_4/q_out/BiasAddBiasAdd(default_policy/sequential_4/q_out/MatMul8default_policy/sequential_4/q_out/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
w
5default_policy/sequential_5/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
ё
0default_policy/sequential_5/concatenate_3/concatConcatV2default_policy/model_out_1default_policy/actions_15default_policy/sequential_5/concatenate_3/concat/axis*
N*
T0*

Tidx0*(
_output_shapes
:џџџџџџџџџ
ы
Sdefault_policy/sequential_5/twin_q_hidden_0/kernel/Initializer/random_uniform/shapeConst*E
_class;
97loc:@default_policy/sequential_5/twin_q_hidden_0/kernel*
_output_shapes
:*
dtype0*
valueB"     
н
Qdefault_policy/sequential_5/twin_q_hidden_0/kernel/Initializer/random_uniform/minConst*E
_class;
97loc:@default_policy/sequential_5/twin_q_hidden_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *К мН
н
Qdefault_policy/sequential_5/twin_q_hidden_0/kernel/Initializer/random_uniform/maxConst*E
_class;
97loc:@default_policy/sequential_5/twin_q_hidden_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *К м=
й
[default_policy/sequential_5/twin_q_hidden_0/kernel/Initializer/random_uniform/RandomUniformRandomUniformSdefault_policy/sequential_5/twin_q_hidden_0/kernel/Initializer/random_uniform/shape*
T0*E
_class;
97loc:@default_policy/sequential_5/twin_q_hidden_0/kernel* 
_output_shapes
:
*
dtype0*

seedd*
seed2
ц
Qdefault_policy/sequential_5/twin_q_hidden_0/kernel/Initializer/random_uniform/subSubQdefault_policy/sequential_5/twin_q_hidden_0/kernel/Initializer/random_uniform/maxQdefault_policy/sequential_5/twin_q_hidden_0/kernel/Initializer/random_uniform/min*
T0*E
_class;
97loc:@default_policy/sequential_5/twin_q_hidden_0/kernel*
_output_shapes
: 
њ
Qdefault_policy/sequential_5/twin_q_hidden_0/kernel/Initializer/random_uniform/mulMul[default_policy/sequential_5/twin_q_hidden_0/kernel/Initializer/random_uniform/RandomUniformQdefault_policy/sequential_5/twin_q_hidden_0/kernel/Initializer/random_uniform/sub*
T0*E
_class;
97loc:@default_policy/sequential_5/twin_q_hidden_0/kernel* 
_output_shapes
:

ю
Mdefault_policy/sequential_5/twin_q_hidden_0/kernel/Initializer/random_uniformAddV2Qdefault_policy/sequential_5/twin_q_hidden_0/kernel/Initializer/random_uniform/mulQdefault_policy/sequential_5/twin_q_hidden_0/kernel/Initializer/random_uniform/min*
T0*E
_class;
97loc:@default_policy/sequential_5/twin_q_hidden_0/kernel* 
_output_shapes
:

Б
2default_policy/sequential_5/twin_q_hidden_0/kernelVarHandleOp*E
_class;
97loc:@default_policy/sequential_5/twin_q_hidden_0/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:
*C
shared_name42default_policy/sequential_5/twin_q_hidden_0/kernel
Е
Sdefault_policy/sequential_5/twin_q_hidden_0/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp2default_policy/sequential_5/twin_q_hidden_0/kernel*
_output_shapes
: 

9default_policy/sequential_5/twin_q_hidden_0/kernel/AssignAssignVariableOp2default_policy/sequential_5/twin_q_hidden_0/kernelMdefault_policy/sequential_5/twin_q_hidden_0/kernel/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Л
Fdefault_policy/sequential_5/twin_q_hidden_0/kernel/Read/ReadVariableOpReadVariableOp2default_policy/sequential_5/twin_q_hidden_0/kernel* 
_output_shapes
:
*
dtype0
ж
Bdefault_policy/sequential_5/twin_q_hidden_0/bias/Initializer/zerosConst*C
_class9
75loc:@default_policy/sequential_5/twin_q_hidden_0/bias*
_output_shapes	
:*
dtype0*
valueB*    
І
0default_policy/sequential_5/twin_q_hidden_0/biasVarHandleOp*C
_class9
75loc:@default_policy/sequential_5/twin_q_hidden_0/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*A
shared_name20default_policy/sequential_5/twin_q_hidden_0/bias
Б
Qdefault_policy/sequential_5/twin_q_hidden_0/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp0default_policy/sequential_5/twin_q_hidden_0/bias*
_output_shapes
: 

7default_policy/sequential_5/twin_q_hidden_0/bias/AssignAssignVariableOp0default_policy/sequential_5/twin_q_hidden_0/biasBdefault_policy/sequential_5/twin_q_hidden_0/bias/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
В
Ddefault_policy/sequential_5/twin_q_hidden_0/bias/Read/ReadVariableOpReadVariableOp0default_policy/sequential_5/twin_q_hidden_0/bias*
_output_shapes	
:*
dtype0
Ж
Adefault_policy/sequential_5/twin_q_hidden_0/MatMul/ReadVariableOpReadVariableOp2default_policy/sequential_5/twin_q_hidden_0/kernel* 
_output_shapes
:
*
dtype0

2default_policy/sequential_5/twin_q_hidden_0/MatMulMatMul0default_policy/sequential_5/concatenate_3/concatAdefault_policy/sequential_5/twin_q_hidden_0/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
А
Bdefault_policy/sequential_5/twin_q_hidden_0/BiasAdd/ReadVariableOpReadVariableOp0default_policy/sequential_5/twin_q_hidden_0/bias*
_output_shapes	
:*
dtype0

3default_policy/sequential_5/twin_q_hidden_0/BiasAddBiasAdd2default_policy/sequential_5/twin_q_hidden_0/MatMulBdefault_policy/sequential_5/twin_q_hidden_0/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
 
0default_policy/sequential_5/twin_q_hidden_0/ReluRelu3default_policy/sequential_5/twin_q_hidden_0/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
ы
Sdefault_policy/sequential_5/twin_q_hidden_1/kernel/Initializer/random_uniform/shapeConst*E
_class;
97loc:@default_policy/sequential_5/twin_q_hidden_1/kernel*
_output_shapes
:*
dtype0*
valueB"      
н
Qdefault_policy/sequential_5/twin_q_hidden_1/kernel/Initializer/random_uniform/minConst*E
_class;
97loc:@default_policy/sequential_5/twin_q_hidden_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *зГнН
н
Qdefault_policy/sequential_5/twin_q_hidden_1/kernel/Initializer/random_uniform/maxConst*E
_class;
97loc:@default_policy/sequential_5/twin_q_hidden_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *зГн=
й
[default_policy/sequential_5/twin_q_hidden_1/kernel/Initializer/random_uniform/RandomUniformRandomUniformSdefault_policy/sequential_5/twin_q_hidden_1/kernel/Initializer/random_uniform/shape*
T0*E
_class;
97loc:@default_policy/sequential_5/twin_q_hidden_1/kernel* 
_output_shapes
:
*
dtype0*

seedd*
seed2
ц
Qdefault_policy/sequential_5/twin_q_hidden_1/kernel/Initializer/random_uniform/subSubQdefault_policy/sequential_5/twin_q_hidden_1/kernel/Initializer/random_uniform/maxQdefault_policy/sequential_5/twin_q_hidden_1/kernel/Initializer/random_uniform/min*
T0*E
_class;
97loc:@default_policy/sequential_5/twin_q_hidden_1/kernel*
_output_shapes
: 
њ
Qdefault_policy/sequential_5/twin_q_hidden_1/kernel/Initializer/random_uniform/mulMul[default_policy/sequential_5/twin_q_hidden_1/kernel/Initializer/random_uniform/RandomUniformQdefault_policy/sequential_5/twin_q_hidden_1/kernel/Initializer/random_uniform/sub*
T0*E
_class;
97loc:@default_policy/sequential_5/twin_q_hidden_1/kernel* 
_output_shapes
:

ю
Mdefault_policy/sequential_5/twin_q_hidden_1/kernel/Initializer/random_uniformAddV2Qdefault_policy/sequential_5/twin_q_hidden_1/kernel/Initializer/random_uniform/mulQdefault_policy/sequential_5/twin_q_hidden_1/kernel/Initializer/random_uniform/min*
T0*E
_class;
97loc:@default_policy/sequential_5/twin_q_hidden_1/kernel* 
_output_shapes
:

Б
2default_policy/sequential_5/twin_q_hidden_1/kernelVarHandleOp*E
_class;
97loc:@default_policy/sequential_5/twin_q_hidden_1/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:
*C
shared_name42default_policy/sequential_5/twin_q_hidden_1/kernel
Е
Sdefault_policy/sequential_5/twin_q_hidden_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp2default_policy/sequential_5/twin_q_hidden_1/kernel*
_output_shapes
: 

9default_policy/sequential_5/twin_q_hidden_1/kernel/AssignAssignVariableOp2default_policy/sequential_5/twin_q_hidden_1/kernelMdefault_policy/sequential_5/twin_q_hidden_1/kernel/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Л
Fdefault_policy/sequential_5/twin_q_hidden_1/kernel/Read/ReadVariableOpReadVariableOp2default_policy/sequential_5/twin_q_hidden_1/kernel* 
_output_shapes
:
*
dtype0
ж
Bdefault_policy/sequential_5/twin_q_hidden_1/bias/Initializer/zerosConst*C
_class9
75loc:@default_policy/sequential_5/twin_q_hidden_1/bias*
_output_shapes	
:*
dtype0*
valueB*    
І
0default_policy/sequential_5/twin_q_hidden_1/biasVarHandleOp*C
_class9
75loc:@default_policy/sequential_5/twin_q_hidden_1/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*A
shared_name20default_policy/sequential_5/twin_q_hidden_1/bias
Б
Qdefault_policy/sequential_5/twin_q_hidden_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp0default_policy/sequential_5/twin_q_hidden_1/bias*
_output_shapes
: 

7default_policy/sequential_5/twin_q_hidden_1/bias/AssignAssignVariableOp0default_policy/sequential_5/twin_q_hidden_1/biasBdefault_policy/sequential_5/twin_q_hidden_1/bias/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
В
Ddefault_policy/sequential_5/twin_q_hidden_1/bias/Read/ReadVariableOpReadVariableOp0default_policy/sequential_5/twin_q_hidden_1/bias*
_output_shapes	
:*
dtype0
Ж
Adefault_policy/sequential_5/twin_q_hidden_1/MatMul/ReadVariableOpReadVariableOp2default_policy/sequential_5/twin_q_hidden_1/kernel* 
_output_shapes
:
*
dtype0

2default_policy/sequential_5/twin_q_hidden_1/MatMulMatMul0default_policy/sequential_5/twin_q_hidden_0/ReluAdefault_policy/sequential_5/twin_q_hidden_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
А
Bdefault_policy/sequential_5/twin_q_hidden_1/BiasAdd/ReadVariableOpReadVariableOp0default_policy/sequential_5/twin_q_hidden_1/bias*
_output_shapes	
:*
dtype0

3default_policy/sequential_5/twin_q_hidden_1/BiasAddBiasAdd2default_policy/sequential_5/twin_q_hidden_1/MatMulBdefault_policy/sequential_5/twin_q_hidden_1/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
 
0default_policy/sequential_5/twin_q_hidden_1/ReluRelu3default_policy/sequential_5/twin_q_hidden_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
с
Ndefault_policy/sequential_5/twin_q_out/kernel/Initializer/random_uniform/shapeConst*@
_class6
42loc:@default_policy/sequential_5/twin_q_out/kernel*
_output_shapes
:*
dtype0*
valueB"      
г
Ldefault_policy/sequential_5/twin_q_out/kernel/Initializer/random_uniform/minConst*@
_class6
42loc:@default_policy/sequential_5/twin_q_out/kernel*
_output_shapes
: *
dtype0*
valueB
 *IvО
г
Ldefault_policy/sequential_5/twin_q_out/kernel/Initializer/random_uniform/maxConst*@
_class6
42loc:@default_policy/sequential_5/twin_q_out/kernel*
_output_shapes
: *
dtype0*
valueB
 *Iv>
Щ
Vdefault_policy/sequential_5/twin_q_out/kernel/Initializer/random_uniform/RandomUniformRandomUniformNdefault_policy/sequential_5/twin_q_out/kernel/Initializer/random_uniform/shape*
T0*@
_class6
42loc:@default_policy/sequential_5/twin_q_out/kernel*
_output_shapes
:	*
dtype0*

seedd*
seed2
в
Ldefault_policy/sequential_5/twin_q_out/kernel/Initializer/random_uniform/subSubLdefault_policy/sequential_5/twin_q_out/kernel/Initializer/random_uniform/maxLdefault_policy/sequential_5/twin_q_out/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@default_policy/sequential_5/twin_q_out/kernel*
_output_shapes
: 
х
Ldefault_policy/sequential_5/twin_q_out/kernel/Initializer/random_uniform/mulMulVdefault_policy/sequential_5/twin_q_out/kernel/Initializer/random_uniform/RandomUniformLdefault_policy/sequential_5/twin_q_out/kernel/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@default_policy/sequential_5/twin_q_out/kernel*
_output_shapes
:	
й
Hdefault_policy/sequential_5/twin_q_out/kernel/Initializer/random_uniformAddV2Ldefault_policy/sequential_5/twin_q_out/kernel/Initializer/random_uniform/mulLdefault_policy/sequential_5/twin_q_out/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@default_policy/sequential_5/twin_q_out/kernel*
_output_shapes
:	
Ё
-default_policy/sequential_5/twin_q_out/kernelVarHandleOp*@
_class6
42loc:@default_policy/sequential_5/twin_q_out/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:	*>
shared_name/-default_policy/sequential_5/twin_q_out/kernel
Ћ
Ndefault_policy/sequential_5/twin_q_out/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp-default_policy/sequential_5/twin_q_out/kernel*
_output_shapes
: 

4default_policy/sequential_5/twin_q_out/kernel/AssignAssignVariableOp-default_policy/sequential_5/twin_q_out/kernelHdefault_policy/sequential_5/twin_q_out/kernel/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
А
Adefault_policy/sequential_5/twin_q_out/kernel/Read/ReadVariableOpReadVariableOp-default_policy/sequential_5/twin_q_out/kernel*
_output_shapes
:	*
dtype0
Ъ
=default_policy/sequential_5/twin_q_out/bias/Initializer/zerosConst*>
_class4
20loc:@default_policy/sequential_5/twin_q_out/bias*
_output_shapes
:*
dtype0*
valueB*    

+default_policy/sequential_5/twin_q_out/biasVarHandleOp*>
_class4
20loc:@default_policy/sequential_5/twin_q_out/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*<
shared_name-+default_policy/sequential_5/twin_q_out/bias
Ї
Ldefault_policy/sequential_5/twin_q_out/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp+default_policy/sequential_5/twin_q_out/bias*
_output_shapes
: 
§
2default_policy/sequential_5/twin_q_out/bias/AssignAssignVariableOp+default_policy/sequential_5/twin_q_out/bias=default_policy/sequential_5/twin_q_out/bias/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Ї
?default_policy/sequential_5/twin_q_out/bias/Read/ReadVariableOpReadVariableOp+default_policy/sequential_5/twin_q_out/bias*
_output_shapes
:*
dtype0
Ћ
<default_policy/sequential_5/twin_q_out/MatMul/ReadVariableOpReadVariableOp-default_policy/sequential_5/twin_q_out/kernel*
_output_shapes
:	*
dtype0
џ
-default_policy/sequential_5/twin_q_out/MatMulMatMul0default_policy/sequential_5/twin_q_hidden_1/Relu<default_policy/sequential_5/twin_q_out/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ѕ
=default_policy/sequential_5/twin_q_out/BiasAdd/ReadVariableOpReadVariableOp+default_policy/sequential_5/twin_q_out/bias*
_output_shapes
:*
dtype0
№
.default_policy/sequential_5/twin_q_out/BiasAddBiasAdd-default_policy/sequential_5/twin_q_out/MatMul=default_policy/sequential_5/twin_q_out/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
Ј
4default_policy/log_alpha_1/Initializer/initial_valueConst*-
_class#
!loc:@default_policy/log_alpha_1*
_output_shapes
: *
dtype0*
valueB
 *    
п
default_policy/log_alpha_1VarHandleOp*-
_class#
!loc:@default_policy/log_alpha_1*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *+
shared_namedefault_policy/log_alpha_1

;default_policy/log_alpha_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/log_alpha_1*
_output_shapes
: 
в
!default_policy/log_alpha_1/AssignAssignVariableOpdefault_policy/log_alpha_14default_policy/log_alpha_1/Initializer/initial_value*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

.default_policy/log_alpha_1/Read/ReadVariableOpReadVariableOpdefault_policy/log_alpha_1*
_output_shapes
: *
dtype0
v
#default_policy/Exp_1/ReadVariableOpReadVariableOpdefault_policy/log_alpha_1*
_output_shapes
: *
dtype0
a
default_policy/Exp_1Exp#default_policy/Exp_1/ReadVariableOp*
T0*
_output_shapes
: 

4default_policy/model/value_out/MatMul/ReadVariableOpReadVariableOpdefault_policy/value_out/kernel*
_output_shapes
:	*
dtype0
й
%default_policy/model/value_out/MatMulMatMuldefault_policy/observation4default_policy/model/value_out/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

5default_policy/model/value_out/BiasAdd/ReadVariableOpReadVariableOpdefault_policy/value_out/bias*
_output_shapes
:*
dtype0
и
&default_policy/model/value_out/BiasAddBiasAdd%default_policy/model/value_out/MatMul5default_policy/model/value_out/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
І
:default_policy/sequential_6/action_1/MatMul/ReadVariableOpReadVariableOp)default_policy/sequential/action_1/kernel* 
_output_shapes
:
*
dtype0
ц
+default_policy/sequential_6/action_1/MatMulMatMuldefault_policy/observation:default_policy/sequential_6/action_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
 
;default_policy/sequential_6/action_1/BiasAdd/ReadVariableOpReadVariableOp'default_policy/sequential/action_1/bias*
_output_shapes	
:*
dtype0
ы
,default_policy/sequential_6/action_1/BiasAddBiasAdd+default_policy/sequential_6/action_1/MatMul;default_policy/sequential_6/action_1/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC

)default_policy/sequential_6/action_1/ReluRelu,default_policy/sequential_6/action_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
І
:default_policy/sequential_6/action_2/MatMul/ReadVariableOpReadVariableOp)default_policy/sequential/action_2/kernel* 
_output_shapes
:
*
dtype0
ѕ
+default_policy/sequential_6/action_2/MatMulMatMul)default_policy/sequential_6/action_1/Relu:default_policy/sequential_6/action_2/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
 
;default_policy/sequential_6/action_2/BiasAdd/ReadVariableOpReadVariableOp'default_policy/sequential/action_2/bias*
_output_shapes	
:*
dtype0
ы
,default_policy/sequential_6/action_2/BiasAddBiasAdd+default_policy/sequential_6/action_2/MatMul;default_policy/sequential_6/action_2/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC

)default_policy/sequential_6/action_2/ReluRelu,default_policy/sequential_6/action_2/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
Љ
<default_policy/sequential_6/action_out/MatMul/ReadVariableOpReadVariableOp+default_policy/sequential/action_out/kernel*
_output_shapes
:	*
dtype0
ј
-default_policy/sequential_6/action_out/MatMulMatMul)default_policy/sequential_6/action_2/Relu<default_policy/sequential_6/action_out/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ѓ
=default_policy/sequential_6/action_out/BiasAdd/ReadVariableOpReadVariableOp)default_policy/sequential/action_out/bias*
_output_shapes
:*
dtype0
№
.default_policy/sequential_6/action_out/BiasAddBiasAdd-default_policy/sequential_6/action_out/MatMul=default_policy/sequential_6/action_out/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
i
default_policy/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
У
default_policy/splitSplitdefault_policy/split/split_dim.default_policy/sequential_6/action_out/BiasAdd*
T0*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*
	num_split
k
&default_policy/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
Ё
$default_policy/clip_by_value/MinimumMinimumdefault_policy/split:1&default_policy/clip_by_value/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
c
default_policy/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   С

default_policy/clip_by_valueMaximum$default_policy/clip_by_value/Minimumdefault_policy/clip_by_value/y*
T0*'
_output_shapes
:џџџџџџџџџ
k
default_policy/Exp_2Expdefault_policy/clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџ
}
:default_policy/default_policy_Normal/sample/sample_shape/xConst*
_output_shapes
: *
dtype0*
valueB 
О
8default_policy/default_policy_Normal/sample/sample_shapeCast:default_policy/default_policy_Normal/sample/sample_shape/x*

DstT0*

SrcT0*
Truncate( *
_output_shapes
: 

1default_policy/default_policy_Normal/sample/ShapeShapedefault_policy/split*
T0*
_output_shapes
:*
out_type0

3default_policy/default_policy_Normal/sample/Shape_1Shapedefault_policy/Exp_2*
T0*
_output_shapes
:*
out_type0
з
9default_policy/default_policy_Normal/sample/BroadcastArgsBroadcastArgs1default_policy/default_policy_Normal/sample/Shape3default_policy/default_policy_Normal/sample/Shape_1*
T0*
_output_shapes
:

;default_policy/default_policy_Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:
y
7default_policy/default_policy_Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Љ
2default_policy/default_policy_Normal/sample/concatConcatV2;default_policy/default_policy_Normal/sample/concat/values_09default_policy/default_policy_Normal/sample/BroadcastArgs7default_policy/default_policy_Normal/sample/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:

Edefault_policy/default_policy_Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    

Gdefault_policy/default_policy_Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?

Udefault_policy/default_policy_Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal2default_policy/default_policy_Normal/sample/concat*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
dtype0*

seedd*
seed2
Њ
Ddefault_policy/default_policy_Normal/sample/normal/random_normal/mulMulUdefault_policy/default_policy_Normal/sample/normal/random_normal/RandomStandardNormalGdefault_policy/default_policy_Normal/sample/normal/random_normal/stddev*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ

@default_policy/default_policy_Normal/sample/normal/random_normalAddV2Ddefault_policy/default_policy_Normal/sample/normal/random_normal/mulEdefault_policy/default_policy_Normal/sample/normal/random_normal/mean*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
Ф
/default_policy/default_policy_Normal/sample/mulMul@default_policy/default_policy_Normal/sample/normal/random_normaldefault_policy/Exp_2*
T0*+
_output_shapes
:џџџџџџџџџ
Е
/default_policy/default_policy_Normal/sample/addAddV2/default_policy/default_policy_Normal/sample/muldefault_policy/split*
T0*+
_output_shapes
:џџџџџџџџџ
Ђ
3default_policy/default_policy_Normal/sample/Shape_2Shape/default_policy/default_policy_Normal/sample/add*
T0*
_output_shapes
:*
out_type0

?default_policy/default_policy_Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:

Adefault_policy/default_policy_Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 

Adefault_policy/default_policy_Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
л
9default_policy/default_policy_Normal/sample/strided_sliceStridedSlice3default_policy/default_policy_Normal/sample/Shape_2?default_policy/default_policy_Normal/sample/strided_slice/stackAdefault_policy/default_policy_Normal/sample/strided_slice/stack_1Adefault_policy/default_policy_Normal/sample/strided_slice/stack_2*
Index0*
T0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
{
9default_policy/default_policy_Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Њ
4default_policy/default_policy_Normal/sample/concat_1ConcatV28default_policy/default_policy_Normal/sample/sample_shape9default_policy/default_policy_Normal/sample/strided_slice9default_policy/default_policy_Normal/sample/concat_1/axis*
N*
T0*

Tidx0*
_output_shapes
:
х
3default_policy/default_policy_Normal/sample/ReshapeReshape/default_policy/default_policy_Normal/sample/add4default_policy/default_policy_Normal/sample/concat_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

default_policy/TanhTanh3default_policy/default_policy_Normal/sample/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
Y
default_policy/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
x
default_policy/addAddV2default_policy/Tanhdefault_policy/add/y*
T0*'
_output_shapes
:џџџџџџџџџ
]
default_policy/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @

default_policy/truedivRealDivdefault_policy/adddefault_policy/truediv/y*
T0*'
_output_shapes
:џџџџџџџџџ
Y
default_policy/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
y
default_policy/mulMuldefault_policy/truedivdefault_policy/mul/y*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П
{
default_policy/add_1AddV2default_policy/muldefault_policy/add_1/y*
T0*'
_output_shapes
:џџџџџџџџџ
m
(default_policy/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
Ѓ
&default_policy/clip_by_value_1/MinimumMinimumdefault_policy/add_1(default_policy/clip_by_value_1/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
e
 default_policy/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П
Ѕ
default_policy/clip_by_value_1Maximum&default_policy/clip_by_value_1/Minimum default_policy/clip_by_value_1/y*
T0*'
_output_shapes
:џџџџџџџџџ
Y
default_policy/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П

default_policy/subSubdefault_policy/clip_by_value_1default_policy/sub/y*
T0*'
_output_shapes
:џџџџџџџџџ
_
default_policy/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @

default_policy/truediv_1RealDivdefault_policy/subdefault_policy/truediv_1/y*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @

default_policy/mul_1Muldefault_policy/truediv_1default_policy/mul_1/y*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
{
default_policy/sub_1Subdefault_policy/mul_1default_policy/sub_1/y*
T0*'
_output_shapes
:џџџџџџџџџ
m
(default_policy/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *яџ?
Ѓ
&default_policy/clip_by_value_2/MinimumMinimumdefault_policy/sub_1(default_policy/clip_by_value_2/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
e
 default_policy/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *яџП
Ѕ
default_policy/clip_by_value_2Maximum&default_policy/clip_by_value_2/Minimum default_policy/clip_by_value_2/y*
T0*'
_output_shapes
:џџџџџџџџџ
o
default_policy/AtanhAtanhdefault_policy/clip_by_value_2*
T0*'
_output_shapes
:џџџџџџџџџ
 
7default_policy/default_policy_Normal_1/log_prob/truedivRealDivdefault_policy/Atanhdefault_policy/Exp_2*
T0*'
_output_shapes
:џџџџџџџџџ
Ђ
9default_policy/default_policy_Normal_1/log_prob/truediv_1RealDivdefault_policy/splitdefault_policy/Exp_2*
T0*'
_output_shapes
:џџџџџџџџџ
ќ
Adefault_policy/default_policy_Normal_1/log_prob/SquaredDifferenceSquaredDifference7default_policy/default_policy_Normal_1/log_prob/truediv9default_policy/default_policy_Normal_1/log_prob/truediv_1*
T0*'
_output_shapes
:џџџџџџџџџ
z
5default_policy/default_policy_Normal_1/log_prob/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   П
ц
3default_policy/default_policy_Normal_1/log_prob/mulMul5default_policy/default_policy_Normal_1/log_prob/mul/xAdefault_policy/default_policy_Normal_1/log_prob/SquaredDifference*
T0*'
_output_shapes
:џџџџџџџџџ
z
5default_policy/default_policy_Normal_1/log_prob/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?k?

3default_policy/default_policy_Normal_1/log_prob/LogLogdefault_policy/Exp_2*
T0*'
_output_shapes
:џџџџџџџџџ
к
3default_policy/default_policy_Normal_1/log_prob/addAddV25default_policy/default_policy_Normal_1/log_prob/Const3default_policy/default_policy_Normal_1/log_prob/Log*
T0*'
_output_shapes
:џџџџџџџџџ
ж
3default_policy/default_policy_Normal_1/log_prob/subSub3default_policy/default_policy_Normal_1/log_prob/mul3default_policy/default_policy_Normal_1/log_prob/add*
T0*'
_output_shapes
:џџџџџџџџџ
m
(default_policy/clip_by_value_3/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ШB
Т
&default_policy/clip_by_value_3/MinimumMinimum3default_policy/default_policy_Normal_1/log_prob/sub(default_policy/clip_by_value_3/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
e
 default_policy/clip_by_value_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ШТ
Ѕ
default_policy/clip_by_value_3Maximum&default_policy/clip_by_value_3/Minimum default_policy/clip_by_value_3/y*
T0*'
_output_shapes
:џџџџџџџџџ
o
$default_policy/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
Њ
default_policy/SumSumdefault_policy/clip_by_value_3$default_policy/Sum/reduction_indices*
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( 
e
default_policy/Tanh_1Tanhdefault_policy/Atanh*
T0*'
_output_shapes
:џџџџџџџџџ
Y
default_policy/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
x
default_policy/powPowdefault_policy/Tanh_1default_policy/pow/y*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
y
default_policy/sub_2Subdefault_policy/sub_2/xdefault_policy/pow*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75
}
default_policy/add_2AddV2default_policy/sub_2default_policy/add_2/y*
T0*'
_output_shapes
:џџџџџџџџџ
a
default_policy/LogLogdefault_policy/add_2*
T0*'
_output_shapes
:џџџџџџџџџ
q
&default_policy/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
Ђ
default_policy/Sum_1Sumdefault_policy/Log&default_policy/Sum_1/reduction_indices*
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( 
s
default_policy/sub_3Subdefault_policy/Sumdefault_policy/Sum_1*
T0*#
_output_shapes
:џџџџџџџџџ

;default_policy/default_policy_Normal_2/mean/ones_like/ShapeShapedefault_policy/Exp_2*
T0*
_output_shapes
:*
out_type0

;default_policy/default_policy_Normal_2/mean/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ћ
5default_policy/default_policy_Normal_2/mean/ones_likeFill;default_policy/default_policy_Normal_2/mean/ones_like/Shape;default_policy/default_policy_Normal_2/mean/ones_like/Const*
T0*'
_output_shapes
:џџџџџџџџџ*

index_type0
Е
/default_policy/default_policy_Normal_2/mean/mulMuldefault_policy/split5default_policy/default_policy_Normal_2/mean/ones_like*
T0*'
_output_shapes
:џџџџџџџџџ

default_policy/Tanh_2Tanh/default_policy/default_policy_Normal_2/mean/mul*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
~
default_policy/add_3AddV2default_policy/Tanh_2default_policy/add_3/y*
T0*'
_output_shapes
:џџџџџџџџџ
_
default_policy/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @

default_policy/truediv_2RealDivdefault_policy/add_3default_policy/truediv_2/y*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @

default_policy/mul_2Muldefault_policy/truediv_2default_policy/mul_2/y*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П
}
default_policy/add_4AddV2default_policy/mul_2default_policy/add_4/y*
T0*'
_output_shapes
:џџџџџџџџџ
m
(default_policy/clip_by_value_4/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
Ѓ
&default_policy/clip_by_value_4/MinimumMinimumdefault_policy/add_4(default_policy/clip_by_value_4/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
e
 default_policy/clip_by_value_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П
Ѕ
default_policy/clip_by_value_4Maximum&default_policy/clip_by_value_4/Minimum default_policy/clip_by_value_4/y*
T0*'
_output_shapes
:џџџџџџџџџ

default_policy/cond/SwitchSwitchdefault_policy/is_exploringdefault_policy/is_exploring*
T0
*
_output_shapes
: : 
g
default_policy/cond/switch_tIdentitydefault_policy/cond/Switch:1*
T0
*
_output_shapes
: 
e
default_policy/cond/switch_fIdentitydefault_policy/cond/Switch*
T0
*
_output_shapes
: 
e
default_policy/cond/pred_idIdentitydefault_policy/is_exploring*
T0
*
_output_shapes
: 
л
default_policy/cond/Switch_1Switchdefault_policy/clip_by_value_1default_policy/cond/pred_id*
T0*1
_class'
%#loc:@default_policy/clip_by_value_1*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ
л
default_policy/cond/Switch_2Switchdefault_policy/clip_by_value_4default_policy/cond/pred_id*
T0*1
_class'
%#loc:@default_policy/clip_by_value_4*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ

default_policy/cond/MergeMergedefault_policy/cond/Switch_2default_policy/cond/Switch_1:1*
N*
T0*)
_output_shapes
:џџџџџџџџџ: 

default_policy/cond_1/SwitchSwitchdefault_policy/is_exploringdefault_policy/is_exploring*
T0
*
_output_shapes
: : 
k
default_policy/cond_1/switch_tIdentitydefault_policy/cond_1/Switch:1*
T0
*
_output_shapes
: 
i
default_policy/cond_1/switch_fIdentitydefault_policy/cond_1/Switch*
T0
*
_output_shapes
: 
g
default_policy/cond_1/pred_idIdentitydefault_policy/is_exploring*
T0
*
_output_shapes
: 
У
default_policy/cond_1/Switch_1Switchdefault_policy/sub_3default_policy/cond_1/pred_id*
T0*'
_class
loc:@default_policy/sub_3*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
}
default_policy/cond_1/ShapeShape"default_policy/cond_1/Shape/Switch*
T0*
_output_shapes
:*
out_type0
й
"default_policy/cond_1/Shape/SwitchSwitchdefault_policy/cond/Mergedefault_policy/cond_1/pred_id*
T0*,
_class"
 loc:@default_policy/cond/Merge*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ

)default_policy/cond_1/strided_slice/stackConst^default_policy/cond_1/switch_f*
_output_shapes
:*
dtype0*
valueB: 

+default_policy/cond_1/strided_slice/stack_1Const^default_policy/cond_1/switch_f*
_output_shapes
:*
dtype0*
valueB:

+default_policy/cond_1/strided_slice/stack_2Const^default_policy/cond_1/switch_f*
_output_shapes
:*
dtype0*
valueB:
ч
#default_policy/cond_1/strided_sliceStridedSlicedefault_policy/cond_1/Shape)default_policy/cond_1/strided_slice/stack+default_policy/cond_1/strided_slice/stack_1+default_policy/cond_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask

"default_policy/cond_1/zeros/packedPack#default_policy/cond_1/strided_slice*
N*
T0*
_output_shapes
:*

axis 

!default_policy/cond_1/zeros/ConstConst^default_policy/cond_1/switch_f*
_output_shapes
: *
dtype0*
valueB
 *    
Њ
default_policy/cond_1/zerosFill"default_policy/cond_1/zeros/packed!default_policy/cond_1/zeros/Const*
T0*#
_output_shapes
:џџџџџџџџџ*

index_type0

default_policy/cond_1/MergeMergedefault_policy/cond_1/zeros default_policy/cond_1/Switch_1:1*
N*
T0*%
_output_shapes
:џџџџџџџџџ: 
f
default_policy/Exp_3Expdefault_policy/cond_1/Merge*
T0*#
_output_shapes
:џџџџџџџџџ
k
 default_policy/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
Ч
default_policy/split_1Split default_policy/split_1/split_dim.default_policy/sequential_6/action_out/BiasAdd*
T0*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*
	num_split
m
(default_policy/clip_by_value_5/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
Ї
&default_policy/clip_by_value_5/MinimumMinimumdefault_policy/split_1:1(default_policy/clip_by_value_5/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
e
 default_policy/clip_by_value_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   С
Ѕ
default_policy/clip_by_value_5Maximum&default_policy/clip_by_value_5/Minimum default_policy/clip_by_value_5/y*
T0*'
_output_shapes
:џџџџџџџџџ
m
default_policy/Exp_4Expdefault_policy/clip_by_value_5*
T0*'
_output_shapes
:џџџџџџџџџ

>default_policy/default_policy_Normal_1_1/sample/sample_shape/xConst*
_output_shapes
: *
dtype0*
valueB 
Ц
<default_policy/default_policy_Normal_1_1/sample/sample_shapeCast>default_policy/default_policy_Normal_1_1/sample/sample_shape/x*

DstT0*

SrcT0*
Truncate( *
_output_shapes
: 

5default_policy/default_policy_Normal_1_1/sample/ShapeShapedefault_policy/split_1*
T0*
_output_shapes
:*
out_type0

7default_policy/default_policy_Normal_1_1/sample/Shape_1Shapedefault_policy/Exp_4*
T0*
_output_shapes
:*
out_type0
у
=default_policy/default_policy_Normal_1_1/sample/BroadcastArgsBroadcastArgs5default_policy/default_policy_Normal_1_1/sample/Shape7default_policy/default_policy_Normal_1_1/sample/Shape_1*
T0*
_output_shapes
:

?default_policy/default_policy_Normal_1_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:
}
;default_policy/default_policy_Normal_1_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Й
6default_policy/default_policy_Normal_1_1/sample/concatConcatV2?default_policy/default_policy_Normal_1_1/sample/concat/values_0=default_policy/default_policy_Normal_1_1/sample/BroadcastArgs;default_policy/default_policy_Normal_1_1/sample/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:

Idefault_policy/default_policy_Normal_1_1/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    

Kdefault_policy/default_policy_Normal_1_1/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?

Ydefault_policy/default_policy_Normal_1_1/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal6default_policy/default_policy_Normal_1_1/sample/concat*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
dtype0*

seedd*
seed2
Ж
Hdefault_policy/default_policy_Normal_1_1/sample/normal/random_normal/mulMulYdefault_policy/default_policy_Normal_1_1/sample/normal/random_normal/RandomStandardNormalKdefault_policy/default_policy_Normal_1_1/sample/normal/random_normal/stddev*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
Ё
Ddefault_policy/default_policy_Normal_1_1/sample/normal/random_normalAddV2Hdefault_policy/default_policy_Normal_1_1/sample/normal/random_normal/mulIdefault_policy/default_policy_Normal_1_1/sample/normal/random_normal/mean*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
Ь
3default_policy/default_policy_Normal_1_1/sample/mulMulDdefault_policy/default_policy_Normal_1_1/sample/normal/random_normaldefault_policy/Exp_4*
T0*+
_output_shapes
:џџџџџџџџџ
П
3default_policy/default_policy_Normal_1_1/sample/addAddV23default_policy/default_policy_Normal_1_1/sample/muldefault_policy/split_1*
T0*+
_output_shapes
:џџџџџџџџџ
Њ
7default_policy/default_policy_Normal_1_1/sample/Shape_2Shape3default_policy/default_policy_Normal_1_1/sample/add*
T0*
_output_shapes
:*
out_type0

Cdefault_policy/default_policy_Normal_1_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:

Edefault_policy/default_policy_Normal_1_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 

Edefault_policy/default_policy_Normal_1_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
я
=default_policy/default_policy_Normal_1_1/sample/strided_sliceStridedSlice7default_policy/default_policy_Normal_1_1/sample/Shape_2Cdefault_policy/default_policy_Normal_1_1/sample/strided_slice/stackEdefault_policy/default_policy_Normal_1_1/sample/strided_slice/stack_1Edefault_policy/default_policy_Normal_1_1/sample/strided_slice/stack_2*
Index0*
T0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 

=default_policy/default_policy_Normal_1_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
К
8default_policy/default_policy_Normal_1_1/sample/concat_1ConcatV2<default_policy/default_policy_Normal_1_1/sample/sample_shape=default_policy/default_policy_Normal_1_1/sample/strided_slice=default_policy/default_policy_Normal_1_1/sample/concat_1/axis*
N*
T0*

Tidx0*
_output_shapes
:
ё
7default_policy/default_policy_Normal_1_1/sample/ReshapeReshape3default_policy/default_policy_Normal_1_1/sample/add8default_policy/default_policy_Normal_1_1/sample/concat_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

default_policy/Tanh_3Tanh7default_policy/default_policy_Normal_1_1/sample/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
~
default_policy/add_5AddV2default_policy/Tanh_3default_policy/add_5/y*
T0*'
_output_shapes
:џџџџџџџџџ
_
default_policy/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @

default_policy/truediv_3RealDivdefault_policy/add_5default_policy/truediv_3/y*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/mul_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @

default_policy/mul_3Muldefault_policy/truediv_3default_policy/mul_3/y*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/add_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П
}
default_policy/add_6AddV2default_policy/mul_3default_policy/add_6/y*
T0*'
_output_shapes
:џџџџџџџџџ
m
(default_policy/clip_by_value_6/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
Ѓ
&default_policy/clip_by_value_6/MinimumMinimumdefault_policy/add_6(default_policy/clip_by_value_6/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
e
 default_policy/clip_by_value_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П
Ѕ
default_policy/clip_by_value_6Maximum&default_policy/clip_by_value_6/Minimum default_policy/clip_by_value_6/y*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/sub_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П

default_policy/sub_4Subdefault_policy/clip_by_value_6default_policy/sub_4/y*
T0*'
_output_shapes
:џџџџџџџџџ
_
default_policy/truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @

default_policy/truediv_4RealDivdefault_policy/sub_4default_policy/truediv_4/y*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @

default_policy/mul_4Muldefault_policy/truediv_4default_policy/mul_4/y*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
{
default_policy/sub_5Subdefault_policy/mul_4default_policy/sub_5/y*
T0*'
_output_shapes
:џџџџџџџџџ
m
(default_policy/clip_by_value_7/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *яџ?
Ѓ
&default_policy/clip_by_value_7/MinimumMinimumdefault_policy/sub_5(default_policy/clip_by_value_7/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
e
 default_policy/clip_by_value_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *яџП
Ѕ
default_policy/clip_by_value_7Maximum&default_policy/clip_by_value_7/Minimum default_policy/clip_by_value_7/y*
T0*'
_output_shapes
:џџџџџџџџџ
q
default_policy/Atanh_1Atanhdefault_policy/clip_by_value_7*
T0*'
_output_shapes
:џџџџџџџџџ
Є
9default_policy/default_policy_Normal_1_2/log_prob/truedivRealDivdefault_policy/Atanh_1default_policy/Exp_4*
T0*'
_output_shapes
:џџџџџџџџџ
І
;default_policy/default_policy_Normal_1_2/log_prob/truediv_1RealDivdefault_policy/split_1default_policy/Exp_4*
T0*'
_output_shapes
:џџџџџџџџџ

Cdefault_policy/default_policy_Normal_1_2/log_prob/SquaredDifferenceSquaredDifference9default_policy/default_policy_Normal_1_2/log_prob/truediv;default_policy/default_policy_Normal_1_2/log_prob/truediv_1*
T0*'
_output_shapes
:џџџџџџџџџ
|
7default_policy/default_policy_Normal_1_2/log_prob/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   П
ь
5default_policy/default_policy_Normal_1_2/log_prob/mulMul7default_policy/default_policy_Normal_1_2/log_prob/mul/xCdefault_policy/default_policy_Normal_1_2/log_prob/SquaredDifference*
T0*'
_output_shapes
:џџџџџџџџџ
|
7default_policy/default_policy_Normal_1_2/log_prob/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?k?

5default_policy/default_policy_Normal_1_2/log_prob/LogLogdefault_policy/Exp_4*
T0*'
_output_shapes
:џџџџџџџџџ
р
5default_policy/default_policy_Normal_1_2/log_prob/addAddV27default_policy/default_policy_Normal_1_2/log_prob/Const5default_policy/default_policy_Normal_1_2/log_prob/Log*
T0*'
_output_shapes
:џџџџџџџџџ
м
5default_policy/default_policy_Normal_1_2/log_prob/subSub5default_policy/default_policy_Normal_1_2/log_prob/mul5default_policy/default_policy_Normal_1_2/log_prob/add*
T0*'
_output_shapes
:џџџџџџџџџ
m
(default_policy/clip_by_value_8/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ШB
Ф
&default_policy/clip_by_value_8/MinimumMinimum5default_policy/default_policy_Normal_1_2/log_prob/sub(default_policy/clip_by_value_8/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
e
 default_policy/clip_by_value_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ШТ
Ѕ
default_policy/clip_by_value_8Maximum&default_policy/clip_by_value_8/Minimum default_policy/clip_by_value_8/y*
T0*'
_output_shapes
:џџџџџџџџџ
q
&default_policy/Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
Ў
default_policy/Sum_2Sumdefault_policy/clip_by_value_8&default_policy/Sum_2/reduction_indices*
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( 
g
default_policy/Tanh_4Tanhdefault_policy/Atanh_1*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
|
default_policy/pow_1Powdefault_policy/Tanh_4default_policy/pow_1/y*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/sub_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
{
default_policy/sub_6Subdefault_policy/sub_6/xdefault_policy/pow_1*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75
}
default_policy/add_7AddV2default_policy/sub_6default_policy/add_7/y*
T0*'
_output_shapes
:џџџџџџџџџ
c
default_policy/Log_1Logdefault_policy/add_7*
T0*'
_output_shapes
:џџџџџџџџџ
q
&default_policy/Sum_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
Є
default_policy/Sum_3Sumdefault_policy/Log_1&default_policy/Sum_3/reduction_indices*
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( 
u
default_policy/sub_7Subdefault_policy/Sum_2default_policy/Sum_3*
T0*#
_output_shapes
:џџџџџџџџџ
[
default_policy/sub_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П
|
default_policy/sub_8Subdefault_policy/actiondefault_policy/sub_8/y*
T0*'
_output_shapes
:џџџџџџџџџ
_
default_policy/truediv_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @

default_policy/truediv_5RealDivdefault_policy/sub_8default_policy/truediv_5/y*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @

default_policy/mul_5Muldefault_policy/truediv_5default_policy/mul_5/y*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/sub_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
{
default_policy/sub_9Subdefault_policy/mul_5default_policy/sub_9/y*
T0*'
_output_shapes
:џџџџџџџџџ
m
(default_policy/clip_by_value_9/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *яџ?
Ѓ
&default_policy/clip_by_value_9/MinimumMinimumdefault_policy/sub_9(default_policy/clip_by_value_9/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
e
 default_policy/clip_by_value_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *яџП
Ѕ
default_policy/clip_by_value_9Maximum&default_policy/clip_by_value_9/Minimum default_policy/clip_by_value_9/y*
T0*'
_output_shapes
:џџџџџџџџџ
q
default_policy/Atanh_2Atanhdefault_policy/clip_by_value_9*
T0*'
_output_shapes
:џџџџџџџџџ
Є
9default_policy/default_policy_Normal_1_3/log_prob/truedivRealDivdefault_policy/Atanh_2default_policy/Exp_4*
T0*'
_output_shapes
:џџџџџџџџџ
І
;default_policy/default_policy_Normal_1_3/log_prob/truediv_1RealDivdefault_policy/split_1default_policy/Exp_4*
T0*'
_output_shapes
:џџџџџџџџџ

Cdefault_policy/default_policy_Normal_1_3/log_prob/SquaredDifferenceSquaredDifference9default_policy/default_policy_Normal_1_3/log_prob/truediv;default_policy/default_policy_Normal_1_3/log_prob/truediv_1*
T0*'
_output_shapes
:џџџџџџџџџ
|
7default_policy/default_policy_Normal_1_3/log_prob/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   П
ь
5default_policy/default_policy_Normal_1_3/log_prob/mulMul7default_policy/default_policy_Normal_1_3/log_prob/mul/xCdefault_policy/default_policy_Normal_1_3/log_prob/SquaredDifference*
T0*'
_output_shapes
:џџџџџџџџџ
|
7default_policy/default_policy_Normal_1_3/log_prob/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?k?

5default_policy/default_policy_Normal_1_3/log_prob/LogLogdefault_policy/Exp_4*
T0*'
_output_shapes
:џџџџџџџџџ
р
5default_policy/default_policy_Normal_1_3/log_prob/addAddV27default_policy/default_policy_Normal_1_3/log_prob/Const5default_policy/default_policy_Normal_1_3/log_prob/Log*
T0*'
_output_shapes
:џџџџџџџџџ
м
5default_policy/default_policy_Normal_1_3/log_prob/subSub5default_policy/default_policy_Normal_1_3/log_prob/mul5default_policy/default_policy_Normal_1_3/log_prob/add*
T0*'
_output_shapes
:џџџџџџџџџ
n
)default_policy/clip_by_value_10/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ШB
Ц
'default_policy/clip_by_value_10/MinimumMinimum5default_policy/default_policy_Normal_1_3/log_prob/sub)default_policy/clip_by_value_10/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
f
!default_policy/clip_by_value_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ШТ
Ј
default_policy/clip_by_value_10Maximum'default_policy/clip_by_value_10/Minimum!default_policy/clip_by_value_10/y*
T0*'
_output_shapes
:џџџџџџџџџ
q
&default_policy/Sum_4/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
Џ
default_policy/Sum_4Sumdefault_policy/clip_by_value_10&default_policy/Sum_4/reduction_indices*
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( 
g
default_policy/Tanh_5Tanhdefault_policy/Atanh_2*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
|
default_policy/pow_2Powdefault_policy/Tanh_5default_policy/pow_2/y*
T0*'
_output_shapes
:џџџџџџџџџ
\
default_policy/sub_10/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
}
default_policy/sub_10Subdefault_policy/sub_10/xdefault_policy/pow_2*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/add_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75
~
default_policy/add_8AddV2default_policy/sub_10default_policy/add_8/y*
T0*'
_output_shapes
:џџџџџџџџџ
c
default_policy/Log_2Logdefault_policy/add_8*
T0*'
_output_shapes
:џџџџџџџџџ
q
&default_policy/Sum_5/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
Є
default_policy/Sum_5Sumdefault_policy/Log_2&default_policy/Sum_5/reduction_indices*
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( 
v
default_policy/sub_11Subdefault_policy/Sum_4default_policy/Sum_5*
T0*#
_output_shapes
:џџџџџџџџџ
z
default_policy/action_1Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
Й
default_policy/initNoOp"^default_policy/global_step/Assign ^default_policy/log_alpha/Assign"^default_policy/log_alpha_1/Assign/^default_policy/sequential/action_1/bias/Assign1^default_policy/sequential/action_1/kernel/Assign/^default_policy/sequential/action_2/bias/Assign1^default_policy/sequential/action_2/kernel/Assign1^default_policy/sequential/action_out/bias/Assign3^default_policy/sequential/action_out/kernel/Assign3^default_policy/sequential_1/q_hidden_0/bias/Assign5^default_policy/sequential_1/q_hidden_0/kernel/Assign3^default_policy/sequential_1/q_hidden_1/bias/Assign5^default_policy/sequential_1/q_hidden_1/kernel/Assign.^default_policy/sequential_1/q_out/bias/Assign0^default_policy/sequential_1/q_out/kernel/Assign8^default_policy/sequential_2/twin_q_hidden_0/bias/Assign:^default_policy/sequential_2/twin_q_hidden_0/kernel/Assign8^default_policy/sequential_2/twin_q_hidden_1/bias/Assign:^default_policy/sequential_2/twin_q_hidden_1/kernel/Assign3^default_policy/sequential_2/twin_q_out/bias/Assign5^default_policy/sequential_2/twin_q_out/kernel/Assign1^default_policy/sequential_3/action_1/bias/Assign3^default_policy/sequential_3/action_1/kernel/Assign1^default_policy/sequential_3/action_2/bias/Assign3^default_policy/sequential_3/action_2/kernel/Assign3^default_policy/sequential_3/action_out/bias/Assign5^default_policy/sequential_3/action_out/kernel/Assign3^default_policy/sequential_4/q_hidden_0/bias/Assign5^default_policy/sequential_4/q_hidden_0/kernel/Assign3^default_policy/sequential_4/q_hidden_1/bias/Assign5^default_policy/sequential_4/q_hidden_1/kernel/Assign.^default_policy/sequential_4/q_out/bias/Assign0^default_policy/sequential_4/q_out/kernel/Assign8^default_policy/sequential_5/twin_q_hidden_0/bias/Assign:^default_policy/sequential_5/twin_q_hidden_0/kernel/Assign8^default_policy/sequential_5/twin_q_hidden_1/bias/Assign:^default_policy/sequential_5/twin_q_hidden_1/kernel/Assign3^default_policy/sequential_5/twin_q_out/bias/Assign5^default_policy/sequential_5/twin_q_out/kernel/Assign%^default_policy/value_out/bias/Assign'^default_policy/value_out/kernel/Assign'^default_policy/value_out_1/bias/Assign)^default_policy/value_out_1/kernel/Assign

6default_policy/model_1/value_out/MatMul/ReadVariableOpReadVariableOpdefault_policy/value_out/kernel*
_output_shapes
:	*
dtype0
н
'default_policy/model_1/value_out/MatMulMatMuldefault_policy/observation6default_policy/model_1/value_out/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

7default_policy/model_1/value_out/BiasAdd/ReadVariableOpReadVariableOpdefault_policy/value_out/bias*
_output_shapes
:*
dtype0
о
(default_policy/model_1/value_out/BiasAddBiasAdd'default_policy/model_1/value_out/MatMul7default_policy/model_1/value_out/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
{
default_policy/new_obsPlaceholder*(
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
o
default_policy/donesPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0
*
shape:џџџџџџџџџ
{
default_policy/actions_2Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
q
default_policy/rewardsPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
u
default_policy/action_probPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
u
default_policy/action_logpPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

!default_policy/action_dist_inputsPlaceholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
q
default_policy/weightsPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

6default_policy/model_2/value_out/MatMul/ReadVariableOpReadVariableOpdefault_policy/value_out/kernel*
_output_shapes
:	*
dtype0
н
'default_policy/model_2/value_out/MatMulMatMuldefault_policy/observation6default_policy/model_2/value_out/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

7default_policy/model_2/value_out/BiasAdd/ReadVariableOpReadVariableOpdefault_policy/value_out/bias*
_output_shapes
:*
dtype0
о
(default_policy/model_2/value_out/BiasAddBiasAdd'default_policy/model_2/value_out/MatMul7default_policy/model_2/value_out/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC

6default_policy/model_3/value_out/MatMul/ReadVariableOpReadVariableOpdefault_policy/value_out/kernel*
_output_shapes
:	*
dtype0
й
'default_policy/model_3/value_out/MatMulMatMuldefault_policy/new_obs6default_policy/model_3/value_out/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

7default_policy/model_3/value_out/BiasAdd/ReadVariableOpReadVariableOpdefault_policy/value_out/bias*
_output_shapes
:*
dtype0
о
(default_policy/model_3/value_out/BiasAddBiasAdd'default_policy/model_3/value_out/MatMul7default_policy/model_3/value_out/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC

8default_policy/model_3_1/value_out/MatMul/ReadVariableOpReadVariableOp!default_policy/value_out_1/kernel*
_output_shapes
:	*
dtype0
н
)default_policy/model_3_1/value_out/MatMulMatMuldefault_policy/new_obs8default_policy/model_3_1/value_out/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

9default_policy/model_3_1/value_out/BiasAdd/ReadVariableOpReadVariableOpdefault_policy/value_out_1/bias*
_output_shapes
:*
dtype0
ф
*default_policy/model_3_1/value_out/BiasAddBiasAdd)default_policy/model_3_1/value_out/MatMul9default_policy/model_3_1/value_out/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
І
:default_policy/sequential_7/action_1/MatMul/ReadVariableOpReadVariableOp)default_policy/sequential/action_1/kernel* 
_output_shapes
:
*
dtype0
ц
+default_policy/sequential_7/action_1/MatMulMatMuldefault_policy/observation:default_policy/sequential_7/action_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
 
;default_policy/sequential_7/action_1/BiasAdd/ReadVariableOpReadVariableOp'default_policy/sequential/action_1/bias*
_output_shapes	
:*
dtype0
ы
,default_policy/sequential_7/action_1/BiasAddBiasAdd+default_policy/sequential_7/action_1/MatMul;default_policy/sequential_7/action_1/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC

)default_policy/sequential_7/action_1/ReluRelu,default_policy/sequential_7/action_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
І
:default_policy/sequential_7/action_2/MatMul/ReadVariableOpReadVariableOp)default_policy/sequential/action_2/kernel* 
_output_shapes
:
*
dtype0
ѕ
+default_policy/sequential_7/action_2/MatMulMatMul)default_policy/sequential_7/action_1/Relu:default_policy/sequential_7/action_2/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
 
;default_policy/sequential_7/action_2/BiasAdd/ReadVariableOpReadVariableOp'default_policy/sequential/action_2/bias*
_output_shapes	
:*
dtype0
ы
,default_policy/sequential_7/action_2/BiasAddBiasAdd+default_policy/sequential_7/action_2/MatMul;default_policy/sequential_7/action_2/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC

)default_policy/sequential_7/action_2/ReluRelu,default_policy/sequential_7/action_2/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
Љ
<default_policy/sequential_7/action_out/MatMul/ReadVariableOpReadVariableOp+default_policy/sequential/action_out/kernel*
_output_shapes
:	*
dtype0
ј
-default_policy/sequential_7/action_out/MatMulMatMul)default_policy/sequential_7/action_2/Relu<default_policy/sequential_7/action_out/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ѓ
=default_policy/sequential_7/action_out/BiasAdd/ReadVariableOpReadVariableOp)default_policy/sequential/action_out/bias*
_output_shapes
:*
dtype0
№
.default_policy/sequential_7/action_out/BiasAddBiasAdd-default_policy/sequential_7/action_out/MatMul=default_policy/sequential_7/action_out/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
k
 default_policy/split_2/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
Ч
default_policy/split_2Split default_policy/split_2/split_dim.default_policy/sequential_7/action_out/BiasAdd*
T0*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*
	num_split
n
)default_policy/clip_by_value_11/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
Љ
'default_policy/clip_by_value_11/MinimumMinimumdefault_policy/split_2:1)default_policy/clip_by_value_11/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
f
!default_policy/clip_by_value_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *   С
Ј
default_policy/clip_by_value_11Maximum'default_policy/clip_by_value_11/Minimum!default_policy/clip_by_value_11/y*
T0*'
_output_shapes
:џџџџџџџџџ
n
default_policy/Exp_5Expdefault_policy/clip_by_value_11*
T0*'
_output_shapes
:џџџџџџџџџ

>default_policy/default_policy_Normal_2_1/sample/sample_shape/xConst*
_output_shapes
: *
dtype0*
valueB 
Ц
<default_policy/default_policy_Normal_2_1/sample/sample_shapeCast>default_policy/default_policy_Normal_2_1/sample/sample_shape/x*

DstT0*

SrcT0*
Truncate( *
_output_shapes
: 

5default_policy/default_policy_Normal_2_1/sample/ShapeShapedefault_policy/split_2*
T0*
_output_shapes
:*
out_type0

7default_policy/default_policy_Normal_2_1/sample/Shape_1Shapedefault_policy/Exp_5*
T0*
_output_shapes
:*
out_type0
у
=default_policy/default_policy_Normal_2_1/sample/BroadcastArgsBroadcastArgs5default_policy/default_policy_Normal_2_1/sample/Shape7default_policy/default_policy_Normal_2_1/sample/Shape_1*
T0*
_output_shapes
:

?default_policy/default_policy_Normal_2_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:
}
;default_policy/default_policy_Normal_2_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Й
6default_policy/default_policy_Normal_2_1/sample/concatConcatV2?default_policy/default_policy_Normal_2_1/sample/concat/values_0=default_policy/default_policy_Normal_2_1/sample/BroadcastArgs;default_policy/default_policy_Normal_2_1/sample/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:

Idefault_policy/default_policy_Normal_2_1/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    

Kdefault_policy/default_policy_Normal_2_1/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?

Ydefault_policy/default_policy_Normal_2_1/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal6default_policy/default_policy_Normal_2_1/sample/concat*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
dtype0*

seedd*
seed2
Ж
Hdefault_policy/default_policy_Normal_2_1/sample/normal/random_normal/mulMulYdefault_policy/default_policy_Normal_2_1/sample/normal/random_normal/RandomStandardNormalKdefault_policy/default_policy_Normal_2_1/sample/normal/random_normal/stddev*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
Ё
Ddefault_policy/default_policy_Normal_2_1/sample/normal/random_normalAddV2Hdefault_policy/default_policy_Normal_2_1/sample/normal/random_normal/mulIdefault_policy/default_policy_Normal_2_1/sample/normal/random_normal/mean*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
Ь
3default_policy/default_policy_Normal_2_1/sample/mulMulDdefault_policy/default_policy_Normal_2_1/sample/normal/random_normaldefault_policy/Exp_5*
T0*+
_output_shapes
:џџџџџџџџџ
П
3default_policy/default_policy_Normal_2_1/sample/addAddV23default_policy/default_policy_Normal_2_1/sample/muldefault_policy/split_2*
T0*+
_output_shapes
:џџџџџџџџџ
Њ
7default_policy/default_policy_Normal_2_1/sample/Shape_2Shape3default_policy/default_policy_Normal_2_1/sample/add*
T0*
_output_shapes
:*
out_type0

Cdefault_policy/default_policy_Normal_2_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:

Edefault_policy/default_policy_Normal_2_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 

Edefault_policy/default_policy_Normal_2_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
я
=default_policy/default_policy_Normal_2_1/sample/strided_sliceStridedSlice7default_policy/default_policy_Normal_2_1/sample/Shape_2Cdefault_policy/default_policy_Normal_2_1/sample/strided_slice/stackEdefault_policy/default_policy_Normal_2_1/sample/strided_slice/stack_1Edefault_policy/default_policy_Normal_2_1/sample/strided_slice/stack_2*
Index0*
T0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 

=default_policy/default_policy_Normal_2_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
К
8default_policy/default_policy_Normal_2_1/sample/concat_1ConcatV2<default_policy/default_policy_Normal_2_1/sample/sample_shape=default_policy/default_policy_Normal_2_1/sample/strided_slice=default_policy/default_policy_Normal_2_1/sample/concat_1/axis*
N*
T0*

Tidx0*
_output_shapes
:
ё
7default_policy/default_policy_Normal_2_1/sample/ReshapeReshape3default_policy/default_policy_Normal_2_1/sample/add8default_policy/default_policy_Normal_2_1/sample/concat_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

default_policy/Tanh_6Tanh7default_policy/default_policy_Normal_2_1/sample/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
~
default_policy/add_9AddV2default_policy/Tanh_6default_policy/add_9/y*
T0*'
_output_shapes
:џџџџџџџџџ
_
default_policy/truediv_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @

default_policy/truediv_6RealDivdefault_policy/add_9default_policy/truediv_6/y*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @

default_policy/mul_6Muldefault_policy/truediv_6default_policy/mul_6/y*
T0*'
_output_shapes
:џџџџџџџџџ
\
default_policy/add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П

default_policy/add_10AddV2default_policy/mul_6default_policy/add_10/y*
T0*'
_output_shapes
:џџџџџџџџџ
n
)default_policy/clip_by_value_12/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
І
'default_policy/clip_by_value_12/MinimumMinimumdefault_policy/add_10)default_policy/clip_by_value_12/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
f
!default_policy/clip_by_value_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П
Ј
default_policy/clip_by_value_12Maximum'default_policy/clip_by_value_12/Minimum!default_policy/clip_by_value_12/y*
T0*'
_output_shapes
:џџџџџџџџџ
\
default_policy/sub_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П

default_policy/sub_12Subdefault_policy/clip_by_value_12default_policy/sub_12/y*
T0*'
_output_shapes
:џџџџџџџџџ
_
default_policy/truediv_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @

default_policy/truediv_7RealDivdefault_policy/sub_12default_policy/truediv_7/y*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/mul_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @

default_policy/mul_7Muldefault_policy/truediv_7default_policy/mul_7/y*
T0*'
_output_shapes
:џџџџџџџџџ
\
default_policy/sub_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
}
default_policy/sub_13Subdefault_policy/mul_7default_policy/sub_13/y*
T0*'
_output_shapes
:џџџџџџџџџ
n
)default_policy/clip_by_value_13/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *яџ?
І
'default_policy/clip_by_value_13/MinimumMinimumdefault_policy/sub_13)default_policy/clip_by_value_13/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
f
!default_policy/clip_by_value_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *яџП
Ј
default_policy/clip_by_value_13Maximum'default_policy/clip_by_value_13/Minimum!default_policy/clip_by_value_13/y*
T0*'
_output_shapes
:џџџџџџџџџ
r
default_policy/Atanh_3Atanhdefault_policy/clip_by_value_13*
T0*'
_output_shapes
:џџџџџџџџџ
Є
9default_policy/default_policy_Normal_2_2/log_prob/truedivRealDivdefault_policy/Atanh_3default_policy/Exp_5*
T0*'
_output_shapes
:џџџџџџџџџ
І
;default_policy/default_policy_Normal_2_2/log_prob/truediv_1RealDivdefault_policy/split_2default_policy/Exp_5*
T0*'
_output_shapes
:џџџџџџџџџ

Cdefault_policy/default_policy_Normal_2_2/log_prob/SquaredDifferenceSquaredDifference9default_policy/default_policy_Normal_2_2/log_prob/truediv;default_policy/default_policy_Normal_2_2/log_prob/truediv_1*
T0*'
_output_shapes
:џџџџџџџџџ
|
7default_policy/default_policy_Normal_2_2/log_prob/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   П
ь
5default_policy/default_policy_Normal_2_2/log_prob/mulMul7default_policy/default_policy_Normal_2_2/log_prob/mul/xCdefault_policy/default_policy_Normal_2_2/log_prob/SquaredDifference*
T0*'
_output_shapes
:џџџџџџџџџ
|
7default_policy/default_policy_Normal_2_2/log_prob/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?k?

5default_policy/default_policy_Normal_2_2/log_prob/LogLogdefault_policy/Exp_5*
T0*'
_output_shapes
:џџџџџџџџџ
р
5default_policy/default_policy_Normal_2_2/log_prob/addAddV27default_policy/default_policy_Normal_2_2/log_prob/Const5default_policy/default_policy_Normal_2_2/log_prob/Log*
T0*'
_output_shapes
:џџџџџџџџџ
м
5default_policy/default_policy_Normal_2_2/log_prob/subSub5default_policy/default_policy_Normal_2_2/log_prob/mul5default_policy/default_policy_Normal_2_2/log_prob/add*
T0*'
_output_shapes
:џџџџџџџџџ
n
)default_policy/clip_by_value_14/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ШB
Ц
'default_policy/clip_by_value_14/MinimumMinimum5default_policy/default_policy_Normal_2_2/log_prob/sub)default_policy/clip_by_value_14/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
f
!default_policy/clip_by_value_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ШТ
Ј
default_policy/clip_by_value_14Maximum'default_policy/clip_by_value_14/Minimum!default_policy/clip_by_value_14/y*
T0*'
_output_shapes
:џџџџџџџџџ
q
&default_policy/Sum_6/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
Џ
default_policy/Sum_6Sumdefault_policy/clip_by_value_14&default_policy/Sum_6/reduction_indices*
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( 
g
default_policy/Tanh_7Tanhdefault_policy/Atanh_3*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
|
default_policy/pow_3Powdefault_policy/Tanh_7default_policy/pow_3/y*
T0*'
_output_shapes
:џџџџџџџџџ
\
default_policy/sub_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
}
default_policy/sub_14Subdefault_policy/sub_14/xdefault_policy/pow_3*
T0*'
_output_shapes
:џџџџџџџџџ
\
default_policy/add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75

default_policy/add_11AddV2default_policy/sub_14default_policy/add_11/y*
T0*'
_output_shapes
:џџџџџџџџџ
d
default_policy/Log_3Logdefault_policy/add_11*
T0*'
_output_shapes
:џџџџџџџџџ
q
&default_policy/Sum_7/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
Є
default_policy/Sum_7Sumdefault_policy/Log_3&default_policy/Sum_7/reduction_indices*
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( 
v
default_policy/sub_15Subdefault_policy/Sum_6default_policy/Sum_7*
T0*#
_output_shapes
:џџџџџџџџџ
\
default_policy/sub_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П

default_policy/sub_16Subdefault_policy/clip_by_value_12default_policy/sub_16/y*
T0*'
_output_shapes
:џџџџџџџџџ
_
default_policy/truediv_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @

default_policy/truediv_8RealDivdefault_policy/sub_16default_policy/truediv_8/y*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/mul_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @

default_policy/mul_8Muldefault_policy/truediv_8default_policy/mul_8/y*
T0*'
_output_shapes
:џџџџџџџџџ
\
default_policy/sub_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
}
default_policy/sub_17Subdefault_policy/mul_8default_policy/sub_17/y*
T0*'
_output_shapes
:џџџџџџџџџ
n
)default_policy/clip_by_value_15/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *яџ?
І
'default_policy/clip_by_value_15/MinimumMinimumdefault_policy/sub_17)default_policy/clip_by_value_15/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
f
!default_policy/clip_by_value_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *яџП
Ј
default_policy/clip_by_value_15Maximum'default_policy/clip_by_value_15/Minimum!default_policy/clip_by_value_15/y*
T0*'
_output_shapes
:џџџџџџџџџ
r
default_policy/Atanh_4Atanhdefault_policy/clip_by_value_15*
T0*'
_output_shapes
:џџџџџџџџџ
Є
9default_policy/default_policy_Normal_2_3/log_prob/truedivRealDivdefault_policy/Atanh_4default_policy/Exp_5*
T0*'
_output_shapes
:џџџџџџџџџ
І
;default_policy/default_policy_Normal_2_3/log_prob/truediv_1RealDivdefault_policy/split_2default_policy/Exp_5*
T0*'
_output_shapes
:џџџџџџџџџ

Cdefault_policy/default_policy_Normal_2_3/log_prob/SquaredDifferenceSquaredDifference9default_policy/default_policy_Normal_2_3/log_prob/truediv;default_policy/default_policy_Normal_2_3/log_prob/truediv_1*
T0*'
_output_shapes
:џџџџџџџџџ
|
7default_policy/default_policy_Normal_2_3/log_prob/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   П
ь
5default_policy/default_policy_Normal_2_3/log_prob/mulMul7default_policy/default_policy_Normal_2_3/log_prob/mul/xCdefault_policy/default_policy_Normal_2_3/log_prob/SquaredDifference*
T0*'
_output_shapes
:џџџџџџџџџ
|
7default_policy/default_policy_Normal_2_3/log_prob/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?k?

5default_policy/default_policy_Normal_2_3/log_prob/LogLogdefault_policy/Exp_5*
T0*'
_output_shapes
:џџџџџџџџџ
р
5default_policy/default_policy_Normal_2_3/log_prob/addAddV27default_policy/default_policy_Normal_2_3/log_prob/Const5default_policy/default_policy_Normal_2_3/log_prob/Log*
T0*'
_output_shapes
:џџџџџџџџџ
м
5default_policy/default_policy_Normal_2_3/log_prob/subSub5default_policy/default_policy_Normal_2_3/log_prob/mul5default_policy/default_policy_Normal_2_3/log_prob/add*
T0*'
_output_shapes
:џџџџџџџџџ
n
)default_policy/clip_by_value_16/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ШB
Ц
'default_policy/clip_by_value_16/MinimumMinimum5default_policy/default_policy_Normal_2_3/log_prob/sub)default_policy/clip_by_value_16/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
f
!default_policy/clip_by_value_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ШТ
Ј
default_policy/clip_by_value_16Maximum'default_policy/clip_by_value_16/Minimum!default_policy/clip_by_value_16/y*
T0*'
_output_shapes
:џџџџџџџџџ
q
&default_policy/Sum_8/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
Џ
default_policy/Sum_8Sumdefault_policy/clip_by_value_16&default_policy/Sum_8/reduction_indices*
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( 
g
default_policy/Tanh_8Tanhdefault_policy/Atanh_4*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/pow_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
|
default_policy/pow_4Powdefault_policy/Tanh_8default_policy/pow_4/y*
T0*'
_output_shapes
:џџџџџџџџџ
\
default_policy/sub_18/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
}
default_policy/sub_18Subdefault_policy/sub_18/xdefault_policy/pow_4*
T0*'
_output_shapes
:џџџџџџџџџ
\
default_policy/add_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75

default_policy/add_12AddV2default_policy/sub_18default_policy/add_12/y*
T0*'
_output_shapes
:џџџџџџџџџ
d
default_policy/Log_4Logdefault_policy/add_12*
T0*'
_output_shapes
:џџџџџџџџџ
q
&default_policy/Sum_9/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
Є
default_policy/Sum_9Sumdefault_policy/Log_4&default_policy/Sum_9/reduction_indices*
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( 
v
default_policy/sub_19Subdefault_policy/Sum_8default_policy/Sum_9*
T0*#
_output_shapes
:џџџџџџџџџ
h
default_policy/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ

default_policy/ExpandDims
ExpandDimsdefault_policy/sub_19default_policy/ExpandDims/dim*
T0*

Tdim0*'
_output_shapes
:џџџџџџџџџ
І
:default_policy/sequential_8/action_1/MatMul/ReadVariableOpReadVariableOp)default_policy/sequential/action_1/kernel* 
_output_shapes
:
*
dtype0
т
+default_policy/sequential_8/action_1/MatMulMatMuldefault_policy/new_obs:default_policy/sequential_8/action_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
 
;default_policy/sequential_8/action_1/BiasAdd/ReadVariableOpReadVariableOp'default_policy/sequential/action_1/bias*
_output_shapes	
:*
dtype0
ы
,default_policy/sequential_8/action_1/BiasAddBiasAdd+default_policy/sequential_8/action_1/MatMul;default_policy/sequential_8/action_1/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC

)default_policy/sequential_8/action_1/ReluRelu,default_policy/sequential_8/action_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
І
:default_policy/sequential_8/action_2/MatMul/ReadVariableOpReadVariableOp)default_policy/sequential/action_2/kernel* 
_output_shapes
:
*
dtype0
ѕ
+default_policy/sequential_8/action_2/MatMulMatMul)default_policy/sequential_8/action_1/Relu:default_policy/sequential_8/action_2/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
 
;default_policy/sequential_8/action_2/BiasAdd/ReadVariableOpReadVariableOp'default_policy/sequential/action_2/bias*
_output_shapes	
:*
dtype0
ы
,default_policy/sequential_8/action_2/BiasAddBiasAdd+default_policy/sequential_8/action_2/MatMul;default_policy/sequential_8/action_2/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC

)default_policy/sequential_8/action_2/ReluRelu,default_policy/sequential_8/action_2/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
Љ
<default_policy/sequential_8/action_out/MatMul/ReadVariableOpReadVariableOp+default_policy/sequential/action_out/kernel*
_output_shapes
:	*
dtype0
ј
-default_policy/sequential_8/action_out/MatMulMatMul)default_policy/sequential_8/action_2/Relu<default_policy/sequential_8/action_out/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ѓ
=default_policy/sequential_8/action_out/BiasAdd/ReadVariableOpReadVariableOp)default_policy/sequential/action_out/bias*
_output_shapes
:*
dtype0
№
.default_policy/sequential_8/action_out/BiasAddBiasAdd-default_policy/sequential_8/action_out/MatMul=default_policy/sequential_8/action_out/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
k
 default_policy/split_3/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
Ч
default_policy/split_3Split default_policy/split_3/split_dim.default_policy/sequential_8/action_out/BiasAdd*
T0*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*
	num_split
n
)default_policy/clip_by_value_17/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
Љ
'default_policy/clip_by_value_17/MinimumMinimumdefault_policy/split_3:1)default_policy/clip_by_value_17/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
f
!default_policy/clip_by_value_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *   С
Ј
default_policy/clip_by_value_17Maximum'default_policy/clip_by_value_17/Minimum!default_policy/clip_by_value_17/y*
T0*'
_output_shapes
:џџџџџџџџџ
n
default_policy/Exp_6Expdefault_policy/clip_by_value_17*
T0*'
_output_shapes
:џџџџџџџџџ

<default_policy/default_policy_Normal_3/sample/sample_shape/xConst*
_output_shapes
: *
dtype0*
valueB 
Т
:default_policy/default_policy_Normal_3/sample/sample_shapeCast<default_policy/default_policy_Normal_3/sample/sample_shape/x*

DstT0*

SrcT0*
Truncate( *
_output_shapes
: 

3default_policy/default_policy_Normal_3/sample/ShapeShapedefault_policy/split_3*
T0*
_output_shapes
:*
out_type0

5default_policy/default_policy_Normal_3/sample/Shape_1Shapedefault_policy/Exp_6*
T0*
_output_shapes
:*
out_type0
н
;default_policy/default_policy_Normal_3/sample/BroadcastArgsBroadcastArgs3default_policy/default_policy_Normal_3/sample/Shape5default_policy/default_policy_Normal_3/sample/Shape_1*
T0*
_output_shapes
:

=default_policy/default_policy_Normal_3/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:
{
9default_policy/default_policy_Normal_3/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Б
4default_policy/default_policy_Normal_3/sample/concatConcatV2=default_policy/default_policy_Normal_3/sample/concat/values_0;default_policy/default_policy_Normal_3/sample/BroadcastArgs9default_policy/default_policy_Normal_3/sample/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:

Gdefault_policy/default_policy_Normal_3/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    

Idefault_policy/default_policy_Normal_3/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?

Wdefault_policy/default_policy_Normal_3/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal4default_policy/default_policy_Normal_3/sample/concat*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
dtype0*

seedd*
seed2
А
Fdefault_policy/default_policy_Normal_3/sample/normal/random_normal/mulMulWdefault_policy/default_policy_Normal_3/sample/normal/random_normal/RandomStandardNormalIdefault_policy/default_policy_Normal_3/sample/normal/random_normal/stddev*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ

Bdefault_policy/default_policy_Normal_3/sample/normal/random_normalAddV2Fdefault_policy/default_policy_Normal_3/sample/normal/random_normal/mulGdefault_policy/default_policy_Normal_3/sample/normal/random_normal/mean*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
Ш
1default_policy/default_policy_Normal_3/sample/mulMulBdefault_policy/default_policy_Normal_3/sample/normal/random_normaldefault_policy/Exp_6*
T0*+
_output_shapes
:џџџџџџџџџ
Л
1default_policy/default_policy_Normal_3/sample/addAddV21default_policy/default_policy_Normal_3/sample/muldefault_policy/split_3*
T0*+
_output_shapes
:џџџџџџџџџ
І
5default_policy/default_policy_Normal_3/sample/Shape_2Shape1default_policy/default_policy_Normal_3/sample/add*
T0*
_output_shapes
:*
out_type0

Adefault_policy/default_policy_Normal_3/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:

Cdefault_policy/default_policy_Normal_3/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 

Cdefault_policy/default_policy_Normal_3/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
х
;default_policy/default_policy_Normal_3/sample/strided_sliceStridedSlice5default_policy/default_policy_Normal_3/sample/Shape_2Adefault_policy/default_policy_Normal_3/sample/strided_slice/stackCdefault_policy/default_policy_Normal_3/sample/strided_slice/stack_1Cdefault_policy/default_policy_Normal_3/sample/strided_slice/stack_2*
Index0*
T0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
}
;default_policy/default_policy_Normal_3/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
В
6default_policy/default_policy_Normal_3/sample/concat_1ConcatV2:default_policy/default_policy_Normal_3/sample/sample_shape;default_policy/default_policy_Normal_3/sample/strided_slice;default_policy/default_policy_Normal_3/sample/concat_1/axis*
N*
T0*

Tidx0*
_output_shapes
:
ы
5default_policy/default_policy_Normal_3/sample/ReshapeReshape1default_policy/default_policy_Normal_3/sample/add6default_policy/default_policy_Normal_3/sample/concat_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

default_policy/Tanh_9Tanh5default_policy/default_policy_Normal_3/sample/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
\
default_policy/add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?

default_policy/add_13AddV2default_policy/Tanh_9default_policy/add_13/y*
T0*'
_output_shapes
:џџџџџџџџџ
_
default_policy/truediv_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @

default_policy/truediv_9RealDivdefault_policy/add_13default_policy/truediv_9/y*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @

default_policy/mul_9Muldefault_policy/truediv_9default_policy/mul_9/y*
T0*'
_output_shapes
:џџџџџџџџџ
\
default_policy/add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П

default_policy/add_14AddV2default_policy/mul_9default_policy/add_14/y*
T0*'
_output_shapes
:џџџџџџџџџ
n
)default_policy/clip_by_value_18/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
І
'default_policy/clip_by_value_18/MinimumMinimumdefault_policy/add_14)default_policy/clip_by_value_18/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
f
!default_policy/clip_by_value_18/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П
Ј
default_policy/clip_by_value_18Maximum'default_policy/clip_by_value_18/Minimum!default_policy/clip_by_value_18/y*
T0*'
_output_shapes
:џџџџџџџџџ
\
default_policy/sub_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П

default_policy/sub_20Subdefault_policy/clip_by_value_18default_policy/sub_20/y*
T0*'
_output_shapes
:џџџџџџџџџ
`
default_policy/truediv_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @

default_policy/truediv_10RealDivdefault_policy/sub_20default_policy/truediv_10/y*
T0*'
_output_shapes
:џџџџџџџџџ
\
default_policy/mul_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @

default_policy/mul_10Muldefault_policy/truediv_10default_policy/mul_10/y*
T0*'
_output_shapes
:џџџџџџџџџ
\
default_policy/sub_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
~
default_policy/sub_21Subdefault_policy/mul_10default_policy/sub_21/y*
T0*'
_output_shapes
:џџџџџџџџџ
n
)default_policy/clip_by_value_19/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *яџ?
І
'default_policy/clip_by_value_19/MinimumMinimumdefault_policy/sub_21)default_policy/clip_by_value_19/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
f
!default_policy/clip_by_value_19/yConst*
_output_shapes
: *
dtype0*
valueB
 *яџП
Ј
default_policy/clip_by_value_19Maximum'default_policy/clip_by_value_19/Minimum!default_policy/clip_by_value_19/y*
T0*'
_output_shapes
:џџџџџџџџџ
r
default_policy/Atanh_5Atanhdefault_policy/clip_by_value_19*
T0*'
_output_shapes
:џџџџџџџџџ
Є
9default_policy/default_policy_Normal_3_1/log_prob/truedivRealDivdefault_policy/Atanh_5default_policy/Exp_6*
T0*'
_output_shapes
:џџџџџџџџџ
І
;default_policy/default_policy_Normal_3_1/log_prob/truediv_1RealDivdefault_policy/split_3default_policy/Exp_6*
T0*'
_output_shapes
:џџџџџџџџџ

Cdefault_policy/default_policy_Normal_3_1/log_prob/SquaredDifferenceSquaredDifference9default_policy/default_policy_Normal_3_1/log_prob/truediv;default_policy/default_policy_Normal_3_1/log_prob/truediv_1*
T0*'
_output_shapes
:џџџџџџџџџ
|
7default_policy/default_policy_Normal_3_1/log_prob/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   П
ь
5default_policy/default_policy_Normal_3_1/log_prob/mulMul7default_policy/default_policy_Normal_3_1/log_prob/mul/xCdefault_policy/default_policy_Normal_3_1/log_prob/SquaredDifference*
T0*'
_output_shapes
:џџџџџџџџџ
|
7default_policy/default_policy_Normal_3_1/log_prob/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?k?

5default_policy/default_policy_Normal_3_1/log_prob/LogLogdefault_policy/Exp_6*
T0*'
_output_shapes
:џџџџџџџџџ
р
5default_policy/default_policy_Normal_3_1/log_prob/addAddV27default_policy/default_policy_Normal_3_1/log_prob/Const5default_policy/default_policy_Normal_3_1/log_prob/Log*
T0*'
_output_shapes
:џџџџџџџџџ
м
5default_policy/default_policy_Normal_3_1/log_prob/subSub5default_policy/default_policy_Normal_3_1/log_prob/mul5default_policy/default_policy_Normal_3_1/log_prob/add*
T0*'
_output_shapes
:џџџџџџџџџ
n
)default_policy/clip_by_value_20/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ШB
Ц
'default_policy/clip_by_value_20/MinimumMinimum5default_policy/default_policy_Normal_3_1/log_prob/sub)default_policy/clip_by_value_20/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
f
!default_policy/clip_by_value_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ШТ
Ј
default_policy/clip_by_value_20Maximum'default_policy/clip_by_value_20/Minimum!default_policy/clip_by_value_20/y*
T0*'
_output_shapes
:џџџџџџџџџ
r
'default_policy/Sum_10/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
Б
default_policy/Sum_10Sumdefault_policy/clip_by_value_20'default_policy/Sum_10/reduction_indices*
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( 
h
default_policy/Tanh_10Tanhdefault_policy/Atanh_5*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/pow_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
}
default_policy/pow_5Powdefault_policy/Tanh_10default_policy/pow_5/y*
T0*'
_output_shapes
:џџџџџџџџџ
\
default_policy/sub_22/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
}
default_policy/sub_22Subdefault_policy/sub_22/xdefault_policy/pow_5*
T0*'
_output_shapes
:џџџџџџџџџ
\
default_policy/add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75

default_policy/add_15AddV2default_policy/sub_22default_policy/add_15/y*
T0*'
_output_shapes
:џџџџџџџџџ
d
default_policy/Log_5Logdefault_policy/add_15*
T0*'
_output_shapes
:џџџџџџџџџ
r
'default_policy/Sum_11/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
І
default_policy/Sum_11Sumdefault_policy/Log_5'default_policy/Sum_11/reduction_indices*
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( 
x
default_policy/sub_23Subdefault_policy/Sum_10default_policy/Sum_11*
T0*#
_output_shapes
:џџџџџџџџџ
\
default_policy/sub_24/yConst*
_output_shapes
: *
dtype0*
valueB
 *  П

default_policy/sub_24Subdefault_policy/clip_by_value_18default_policy/sub_24/y*
T0*'
_output_shapes
:џџџџџџџџџ
`
default_policy/truediv_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @

default_policy/truediv_11RealDivdefault_policy/sub_24default_policy/truediv_11/y*
T0*'
_output_shapes
:џџџџџџџџџ
\
default_policy/mul_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @

default_policy/mul_11Muldefault_policy/truediv_11default_policy/mul_11/y*
T0*'
_output_shapes
:џџџџџџџџџ
\
default_policy/sub_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
~
default_policy/sub_25Subdefault_policy/mul_11default_policy/sub_25/y*
T0*'
_output_shapes
:џџџџџџџџџ
n
)default_policy/clip_by_value_21/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *яџ?
І
'default_policy/clip_by_value_21/MinimumMinimumdefault_policy/sub_25)default_policy/clip_by_value_21/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
f
!default_policy/clip_by_value_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *яџП
Ј
default_policy/clip_by_value_21Maximum'default_policy/clip_by_value_21/Minimum!default_policy/clip_by_value_21/y*
T0*'
_output_shapes
:џџџџџџџџџ
r
default_policy/Atanh_6Atanhdefault_policy/clip_by_value_21*
T0*'
_output_shapes
:џџџџџџџџџ
Є
9default_policy/default_policy_Normal_3_2/log_prob/truedivRealDivdefault_policy/Atanh_6default_policy/Exp_6*
T0*'
_output_shapes
:џџџџџџџџџ
І
;default_policy/default_policy_Normal_3_2/log_prob/truediv_1RealDivdefault_policy/split_3default_policy/Exp_6*
T0*'
_output_shapes
:џџџџџџџџџ

Cdefault_policy/default_policy_Normal_3_2/log_prob/SquaredDifferenceSquaredDifference9default_policy/default_policy_Normal_3_2/log_prob/truediv;default_policy/default_policy_Normal_3_2/log_prob/truediv_1*
T0*'
_output_shapes
:џџџџџџџџџ
|
7default_policy/default_policy_Normal_3_2/log_prob/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   П
ь
5default_policy/default_policy_Normal_3_2/log_prob/mulMul7default_policy/default_policy_Normal_3_2/log_prob/mul/xCdefault_policy/default_policy_Normal_3_2/log_prob/SquaredDifference*
T0*'
_output_shapes
:џџџџџџџџџ
|
7default_policy/default_policy_Normal_3_2/log_prob/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?k?

5default_policy/default_policy_Normal_3_2/log_prob/LogLogdefault_policy/Exp_6*
T0*'
_output_shapes
:џџџџџџџџџ
р
5default_policy/default_policy_Normal_3_2/log_prob/addAddV27default_policy/default_policy_Normal_3_2/log_prob/Const5default_policy/default_policy_Normal_3_2/log_prob/Log*
T0*'
_output_shapes
:џџџџџџџџџ
м
5default_policy/default_policy_Normal_3_2/log_prob/subSub5default_policy/default_policy_Normal_3_2/log_prob/mul5default_policy/default_policy_Normal_3_2/log_prob/add*
T0*'
_output_shapes
:џџџџџџџџџ
n
)default_policy/clip_by_value_22/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ШB
Ц
'default_policy/clip_by_value_22/MinimumMinimum5default_policy/default_policy_Normal_3_2/log_prob/sub)default_policy/clip_by_value_22/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
f
!default_policy/clip_by_value_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ШТ
Ј
default_policy/clip_by_value_22Maximum'default_policy/clip_by_value_22/Minimum!default_policy/clip_by_value_22/y*
T0*'
_output_shapes
:џџџџџџџџџ
r
'default_policy/Sum_12/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
Б
default_policy/Sum_12Sumdefault_policy/clip_by_value_22'default_policy/Sum_12/reduction_indices*
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( 
h
default_policy/Tanh_11Tanhdefault_policy/Atanh_6*
T0*'
_output_shapes
:џџџџџџџџџ
[
default_policy/pow_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
}
default_policy/pow_6Powdefault_policy/Tanh_11default_policy/pow_6/y*
T0*'
_output_shapes
:џџџџџџџџџ
\
default_policy/sub_26/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
}
default_policy/sub_26Subdefault_policy/sub_26/xdefault_policy/pow_6*
T0*'
_output_shapes
:џџџџџџџџџ
\
default_policy/add_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75

default_policy/add_16AddV2default_policy/sub_26default_policy/add_16/y*
T0*'
_output_shapes
:џџџџџџџџџ
d
default_policy/Log_6Logdefault_policy/add_16*
T0*'
_output_shapes
:џџџџџџџџџ
r
'default_policy/Sum_13/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
І
default_policy/Sum_13Sumdefault_policy/Log_6'default_policy/Sum_13/reduction_indices*
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( 
x
default_policy/sub_27Subdefault_policy/Sum_12default_policy/Sum_13*
T0*#
_output_shapes
:џџџџџџџџџ
j
default_policy/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ

default_policy/ExpandDims_1
ExpandDimsdefault_policy/sub_27default_policy/ExpandDims_1/dim*
T0*

Tdim0*'
_output_shapes
:џџџџџџџџџ

=default_policy/model_1_1/sequential_1/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :

8default_policy/model_1_1/sequential_1/concatenate/concatConcatV2default_policy/observationdefault_policy/actions_2=default_policy/model_1_1/sequential_1/concatenate/concat/axis*
N*
T0*

Tidx0*(
_output_shapes
:џџџџџџџџџ
Ж
Fdefault_policy/model_1_1/sequential_1/q_hidden_0/MatMul/ReadVariableOpReadVariableOp-default_policy/sequential_1/q_hidden_0/kernel* 
_output_shapes
:
*
dtype0

7default_policy/model_1_1/sequential_1/q_hidden_0/MatMulMatMul8default_policy/model_1_1/sequential_1/concatenate/concatFdefault_policy/model_1_1/sequential_1/q_hidden_0/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
А
Gdefault_policy/model_1_1/sequential_1/q_hidden_0/BiasAdd/ReadVariableOpReadVariableOp+default_policy/sequential_1/q_hidden_0/bias*
_output_shapes	
:*
dtype0

8default_policy/model_1_1/sequential_1/q_hidden_0/BiasAddBiasAdd7default_policy/model_1_1/sequential_1/q_hidden_0/MatMulGdefault_policy/model_1_1/sequential_1/q_hidden_0/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
Њ
5default_policy/model_1_1/sequential_1/q_hidden_0/ReluRelu8default_policy/model_1_1/sequential_1/q_hidden_0/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
Ж
Fdefault_policy/model_1_1/sequential_1/q_hidden_1/MatMul/ReadVariableOpReadVariableOp-default_policy/sequential_1/q_hidden_1/kernel* 
_output_shapes
:
*
dtype0

7default_policy/model_1_1/sequential_1/q_hidden_1/MatMulMatMul5default_policy/model_1_1/sequential_1/q_hidden_0/ReluFdefault_policy/model_1_1/sequential_1/q_hidden_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
А
Gdefault_policy/model_1_1/sequential_1/q_hidden_1/BiasAdd/ReadVariableOpReadVariableOp+default_policy/sequential_1/q_hidden_1/bias*
_output_shapes	
:*
dtype0

8default_policy/model_1_1/sequential_1/q_hidden_1/BiasAddBiasAdd7default_policy/model_1_1/sequential_1/q_hidden_1/MatMulGdefault_policy/model_1_1/sequential_1/q_hidden_1/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
Њ
5default_policy/model_1_1/sequential_1/q_hidden_1/ReluRelu8default_policy/model_1_1/sequential_1/q_hidden_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
Ћ
Adefault_policy/model_1_1/sequential_1/q_out/MatMul/ReadVariableOpReadVariableOp(default_policy/sequential_1/q_out/kernel*
_output_shapes
:	*
dtype0

2default_policy/model_1_1/sequential_1/q_out/MatMulMatMul5default_policy/model_1_1/sequential_1/q_hidden_1/ReluAdefault_policy/model_1_1/sequential_1/q_out/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ѕ
Bdefault_policy/model_1_1/sequential_1/q_out/BiasAdd/ReadVariableOpReadVariableOp&default_policy/sequential_1/q_out/bias*
_output_shapes
:*
dtype0
џ
3default_policy/model_1_1/sequential_1/q_out/BiasAddBiasAdd2default_policy/model_1_1/sequential_1/q_out/MatMulBdefault_policy/model_1_1/sequential_1/q_out/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC

?default_policy/model_2_1/sequential_2/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :

:default_policy/model_2_1/sequential_2/concatenate_1/concatConcatV2default_policy/observationdefault_policy/actions_2?default_policy/model_2_1/sequential_2/concatenate_1/concat/axis*
N*
T0*

Tidx0*(
_output_shapes
:џџџџџџџџџ
Р
Kdefault_policy/model_2_1/sequential_2/twin_q_hidden_0/MatMul/ReadVariableOpReadVariableOp2default_policy/sequential_2/twin_q_hidden_0/kernel* 
_output_shapes
:
*
dtype0
Ј
<default_policy/model_2_1/sequential_2/twin_q_hidden_0/MatMulMatMul:default_policy/model_2_1/sequential_2/concatenate_1/concatKdefault_policy/model_2_1/sequential_2/twin_q_hidden_0/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
К
Ldefault_policy/model_2_1/sequential_2/twin_q_hidden_0/BiasAdd/ReadVariableOpReadVariableOp0default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes	
:*
dtype0

=default_policy/model_2_1/sequential_2/twin_q_hidden_0/BiasAddBiasAdd<default_policy/model_2_1/sequential_2/twin_q_hidden_0/MatMulLdefault_policy/model_2_1/sequential_2/twin_q_hidden_0/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
Д
:default_policy/model_2_1/sequential_2/twin_q_hidden_0/ReluRelu=default_policy/model_2_1/sequential_2/twin_q_hidden_0/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
Р
Kdefault_policy/model_2_1/sequential_2/twin_q_hidden_1/MatMul/ReadVariableOpReadVariableOp2default_policy/sequential_2/twin_q_hidden_1/kernel* 
_output_shapes
:
*
dtype0
Ј
<default_policy/model_2_1/sequential_2/twin_q_hidden_1/MatMulMatMul:default_policy/model_2_1/sequential_2/twin_q_hidden_0/ReluKdefault_policy/model_2_1/sequential_2/twin_q_hidden_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
К
Ldefault_policy/model_2_1/sequential_2/twin_q_hidden_1/BiasAdd/ReadVariableOpReadVariableOp0default_policy/sequential_2/twin_q_hidden_1/bias*
_output_shapes	
:*
dtype0

=default_policy/model_2_1/sequential_2/twin_q_hidden_1/BiasAddBiasAdd<default_policy/model_2_1/sequential_2/twin_q_hidden_1/MatMulLdefault_policy/model_2_1/sequential_2/twin_q_hidden_1/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
Д
:default_policy/model_2_1/sequential_2/twin_q_hidden_1/ReluRelu=default_policy/model_2_1/sequential_2/twin_q_hidden_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
Е
Fdefault_policy/model_2_1/sequential_2/twin_q_out/MatMul/ReadVariableOpReadVariableOp-default_policy/sequential_2/twin_q_out/kernel*
_output_shapes
:	*
dtype0

7default_policy/model_2_1/sequential_2/twin_q_out/MatMulMatMul:default_policy/model_2_1/sequential_2/twin_q_hidden_1/ReluFdefault_policy/model_2_1/sequential_2/twin_q_out/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Џ
Gdefault_policy/model_2_1/sequential_2/twin_q_out/BiasAdd/ReadVariableOpReadVariableOp+default_policy/sequential_2/twin_q_out/bias*
_output_shapes
:*
dtype0

8default_policy/model_2_1/sequential_2/twin_q_out/BiasAddBiasAdd7default_policy/model_2_1/sequential_2/twin_q_out/MatMulGdefault_policy/model_2_1/sequential_2/twin_q_out/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC

=default_policy/model_1_2/sequential_1/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :

8default_policy/model_1_2/sequential_1/concatenate/concatConcatV2default_policy/observationdefault_policy/clip_by_value_12=default_policy/model_1_2/sequential_1/concatenate/concat/axis*
N*
T0*

Tidx0*(
_output_shapes
:џџџџџџџџџ
Ж
Fdefault_policy/model_1_2/sequential_1/q_hidden_0/MatMul/ReadVariableOpReadVariableOp-default_policy/sequential_1/q_hidden_0/kernel* 
_output_shapes
:
*
dtype0

7default_policy/model_1_2/sequential_1/q_hidden_0/MatMulMatMul8default_policy/model_1_2/sequential_1/concatenate/concatFdefault_policy/model_1_2/sequential_1/q_hidden_0/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
А
Gdefault_policy/model_1_2/sequential_1/q_hidden_0/BiasAdd/ReadVariableOpReadVariableOp+default_policy/sequential_1/q_hidden_0/bias*
_output_shapes	
:*
dtype0

8default_policy/model_1_2/sequential_1/q_hidden_0/BiasAddBiasAdd7default_policy/model_1_2/sequential_1/q_hidden_0/MatMulGdefault_policy/model_1_2/sequential_1/q_hidden_0/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
Њ
5default_policy/model_1_2/sequential_1/q_hidden_0/ReluRelu8default_policy/model_1_2/sequential_1/q_hidden_0/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
Ж
Fdefault_policy/model_1_2/sequential_1/q_hidden_1/MatMul/ReadVariableOpReadVariableOp-default_policy/sequential_1/q_hidden_1/kernel* 
_output_shapes
:
*
dtype0

7default_policy/model_1_2/sequential_1/q_hidden_1/MatMulMatMul5default_policy/model_1_2/sequential_1/q_hidden_0/ReluFdefault_policy/model_1_2/sequential_1/q_hidden_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
А
Gdefault_policy/model_1_2/sequential_1/q_hidden_1/BiasAdd/ReadVariableOpReadVariableOp+default_policy/sequential_1/q_hidden_1/bias*
_output_shapes	
:*
dtype0

8default_policy/model_1_2/sequential_1/q_hidden_1/BiasAddBiasAdd7default_policy/model_1_2/sequential_1/q_hidden_1/MatMulGdefault_policy/model_1_2/sequential_1/q_hidden_1/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
Њ
5default_policy/model_1_2/sequential_1/q_hidden_1/ReluRelu8default_policy/model_1_2/sequential_1/q_hidden_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
Ћ
Adefault_policy/model_1_2/sequential_1/q_out/MatMul/ReadVariableOpReadVariableOp(default_policy/sequential_1/q_out/kernel*
_output_shapes
:	*
dtype0

2default_policy/model_1_2/sequential_1/q_out/MatMulMatMul5default_policy/model_1_2/sequential_1/q_hidden_1/ReluAdefault_policy/model_1_2/sequential_1/q_out/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ѕ
Bdefault_policy/model_1_2/sequential_1/q_out/BiasAdd/ReadVariableOpReadVariableOp&default_policy/sequential_1/q_out/bias*
_output_shapes
:*
dtype0
џ
3default_policy/model_1_2/sequential_1/q_out/BiasAddBiasAdd2default_policy/model_1_2/sequential_1/q_out/MatMulBdefault_policy/model_1_2/sequential_1/q_out/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC

?default_policy/model_2_2/sequential_2/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :

:default_policy/model_2_2/sequential_2/concatenate_1/concatConcatV2default_policy/observationdefault_policy/clip_by_value_12?default_policy/model_2_2/sequential_2/concatenate_1/concat/axis*
N*
T0*

Tidx0*(
_output_shapes
:џџџџџџџџџ
Р
Kdefault_policy/model_2_2/sequential_2/twin_q_hidden_0/MatMul/ReadVariableOpReadVariableOp2default_policy/sequential_2/twin_q_hidden_0/kernel* 
_output_shapes
:
*
dtype0
Ј
<default_policy/model_2_2/sequential_2/twin_q_hidden_0/MatMulMatMul:default_policy/model_2_2/sequential_2/concatenate_1/concatKdefault_policy/model_2_2/sequential_2/twin_q_hidden_0/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
К
Ldefault_policy/model_2_2/sequential_2/twin_q_hidden_0/BiasAdd/ReadVariableOpReadVariableOp0default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes	
:*
dtype0

=default_policy/model_2_2/sequential_2/twin_q_hidden_0/BiasAddBiasAdd<default_policy/model_2_2/sequential_2/twin_q_hidden_0/MatMulLdefault_policy/model_2_2/sequential_2/twin_q_hidden_0/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
Д
:default_policy/model_2_2/sequential_2/twin_q_hidden_0/ReluRelu=default_policy/model_2_2/sequential_2/twin_q_hidden_0/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
Р
Kdefault_policy/model_2_2/sequential_2/twin_q_hidden_1/MatMul/ReadVariableOpReadVariableOp2default_policy/sequential_2/twin_q_hidden_1/kernel* 
_output_shapes
:
*
dtype0
Ј
<default_policy/model_2_2/sequential_2/twin_q_hidden_1/MatMulMatMul:default_policy/model_2_2/sequential_2/twin_q_hidden_0/ReluKdefault_policy/model_2_2/sequential_2/twin_q_hidden_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
К
Ldefault_policy/model_2_2/sequential_2/twin_q_hidden_1/BiasAdd/ReadVariableOpReadVariableOp0default_policy/sequential_2/twin_q_hidden_1/bias*
_output_shapes	
:*
dtype0

=default_policy/model_2_2/sequential_2/twin_q_hidden_1/BiasAddBiasAdd<default_policy/model_2_2/sequential_2/twin_q_hidden_1/MatMulLdefault_policy/model_2_2/sequential_2/twin_q_hidden_1/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
Д
:default_policy/model_2_2/sequential_2/twin_q_hidden_1/ReluRelu=default_policy/model_2_2/sequential_2/twin_q_hidden_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
Е
Fdefault_policy/model_2_2/sequential_2/twin_q_out/MatMul/ReadVariableOpReadVariableOp-default_policy/sequential_2/twin_q_out/kernel*
_output_shapes
:	*
dtype0

7default_policy/model_2_2/sequential_2/twin_q_out/MatMulMatMul:default_policy/model_2_2/sequential_2/twin_q_hidden_1/ReluFdefault_policy/model_2_2/sequential_2/twin_q_out/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Џ
Gdefault_policy/model_2_2/sequential_2/twin_q_out/BiasAdd/ReadVariableOpReadVariableOp+default_policy/sequential_2/twin_q_out/bias*
_output_shapes
:*
dtype0

8default_policy/model_2_2/sequential_2/twin_q_out/BiasAddBiasAdd7default_policy/model_2_2/sequential_2/twin_q_out/MatMulGdefault_policy/model_2_2/sequential_2/twin_q_out/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
к
default_policy/Min/inputPack3default_policy/model_1_2/sequential_1/q_out/BiasAdd8default_policy/model_2_2/sequential_2/twin_q_out/BiasAdd*
N*
T0*+
_output_shapes
:џџџџџџџџџ*

axis 
f
$default_policy/Min/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 
Ј
default_policy/MinMindefault_policy/Min/input$default_policy/Min/reduction_indices*
T0*

Tidx0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims( 

=default_policy/model_4/sequential_4/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :

8default_policy/model_4/sequential_4/concatenate_2/concatConcatV2default_policy/new_obsdefault_policy/clip_by_value_18=default_policy/model_4/sequential_4/concatenate_2/concat/axis*
N*
T0*

Tidx0*(
_output_shapes
:џџџџџџџџџ
Д
Ddefault_policy/model_4/sequential_4/q_hidden_0/MatMul/ReadVariableOpReadVariableOp-default_policy/sequential_4/q_hidden_0/kernel* 
_output_shapes
:
*
dtype0

5default_policy/model_4/sequential_4/q_hidden_0/MatMulMatMul8default_policy/model_4/sequential_4/concatenate_2/concatDdefault_policy/model_4/sequential_4/q_hidden_0/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ў
Edefault_policy/model_4/sequential_4/q_hidden_0/BiasAdd/ReadVariableOpReadVariableOp+default_policy/sequential_4/q_hidden_0/bias*
_output_shapes	
:*
dtype0

6default_policy/model_4/sequential_4/q_hidden_0/BiasAddBiasAdd5default_policy/model_4/sequential_4/q_hidden_0/MatMulEdefault_policy/model_4/sequential_4/q_hidden_0/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
І
3default_policy/model_4/sequential_4/q_hidden_0/ReluRelu6default_policy/model_4/sequential_4/q_hidden_0/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
Д
Ddefault_policy/model_4/sequential_4/q_hidden_1/MatMul/ReadVariableOpReadVariableOp-default_policy/sequential_4/q_hidden_1/kernel* 
_output_shapes
:
*
dtype0

5default_policy/model_4/sequential_4/q_hidden_1/MatMulMatMul3default_policy/model_4/sequential_4/q_hidden_0/ReluDdefault_policy/model_4/sequential_4/q_hidden_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ў
Edefault_policy/model_4/sequential_4/q_hidden_1/BiasAdd/ReadVariableOpReadVariableOp+default_policy/sequential_4/q_hidden_1/bias*
_output_shapes	
:*
dtype0

6default_policy/model_4/sequential_4/q_hidden_1/BiasAddBiasAdd5default_policy/model_4/sequential_4/q_hidden_1/MatMulEdefault_policy/model_4/sequential_4/q_hidden_1/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
І
3default_policy/model_4/sequential_4/q_hidden_1/ReluRelu6default_policy/model_4/sequential_4/q_hidden_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
Љ
?default_policy/model_4/sequential_4/q_out/MatMul/ReadVariableOpReadVariableOp(default_policy/sequential_4/q_out/kernel*
_output_shapes
:	*
dtype0

0default_policy/model_4/sequential_4/q_out/MatMulMatMul3default_policy/model_4/sequential_4/q_hidden_1/Relu?default_policy/model_4/sequential_4/q_out/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ѓ
@default_policy/model_4/sequential_4/q_out/BiasAdd/ReadVariableOpReadVariableOp&default_policy/sequential_4/q_out/bias*
_output_shapes
:*
dtype0
љ
1default_policy/model_4/sequential_4/q_out/BiasAddBiasAdd0default_policy/model_4/sequential_4/q_out/MatMul@default_policy/model_4/sequential_4/q_out/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC

=default_policy/model_5/sequential_5/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :

8default_policy/model_5/sequential_5/concatenate_3/concatConcatV2default_policy/new_obsdefault_policy/clip_by_value_18=default_policy/model_5/sequential_5/concatenate_3/concat/axis*
N*
T0*

Tidx0*(
_output_shapes
:џџџџџџџџџ
О
Idefault_policy/model_5/sequential_5/twin_q_hidden_0/MatMul/ReadVariableOpReadVariableOp2default_policy/sequential_5/twin_q_hidden_0/kernel* 
_output_shapes
:
*
dtype0
Ђ
:default_policy/model_5/sequential_5/twin_q_hidden_0/MatMulMatMul8default_policy/model_5/sequential_5/concatenate_3/concatIdefault_policy/model_5/sequential_5/twin_q_hidden_0/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
И
Jdefault_policy/model_5/sequential_5/twin_q_hidden_0/BiasAdd/ReadVariableOpReadVariableOp0default_policy/sequential_5/twin_q_hidden_0/bias*
_output_shapes	
:*
dtype0

;default_policy/model_5/sequential_5/twin_q_hidden_0/BiasAddBiasAdd:default_policy/model_5/sequential_5/twin_q_hidden_0/MatMulJdefault_policy/model_5/sequential_5/twin_q_hidden_0/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
А
8default_policy/model_5/sequential_5/twin_q_hidden_0/ReluRelu;default_policy/model_5/sequential_5/twin_q_hidden_0/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
О
Idefault_policy/model_5/sequential_5/twin_q_hidden_1/MatMul/ReadVariableOpReadVariableOp2default_policy/sequential_5/twin_q_hidden_1/kernel* 
_output_shapes
:
*
dtype0
Ђ
:default_policy/model_5/sequential_5/twin_q_hidden_1/MatMulMatMul8default_policy/model_5/sequential_5/twin_q_hidden_0/ReluIdefault_policy/model_5/sequential_5/twin_q_hidden_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
И
Jdefault_policy/model_5/sequential_5/twin_q_hidden_1/BiasAdd/ReadVariableOpReadVariableOp0default_policy/sequential_5/twin_q_hidden_1/bias*
_output_shapes	
:*
dtype0

;default_policy/model_5/sequential_5/twin_q_hidden_1/BiasAddBiasAdd:default_policy/model_5/sequential_5/twin_q_hidden_1/MatMulJdefault_policy/model_5/sequential_5/twin_q_hidden_1/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
А
8default_policy/model_5/sequential_5/twin_q_hidden_1/ReluRelu;default_policy/model_5/sequential_5/twin_q_hidden_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
Г
Ddefault_policy/model_5/sequential_5/twin_q_out/MatMul/ReadVariableOpReadVariableOp-default_policy/sequential_5/twin_q_out/kernel*
_output_shapes
:	*
dtype0

5default_policy/model_5/sequential_5/twin_q_out/MatMulMatMul8default_policy/model_5/sequential_5/twin_q_hidden_1/ReluDdefault_policy/model_5/sequential_5/twin_q_out/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
­
Edefault_policy/model_5/sequential_5/twin_q_out/BiasAdd/ReadVariableOpReadVariableOp+default_policy/sequential_5/twin_q_out/bias*
_output_shapes
:*
dtype0

6default_policy/model_5/sequential_5/twin_q_out/BiasAddBiasAdd5default_policy/model_5/sequential_5/twin_q_out/MatMulEdefault_policy/model_5/sequential_5/twin_q_out/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC
и
default_policy/Min_1/inputPack1default_policy/model_4/sequential_4/q_out/BiasAdd6default_policy/model_5/sequential_5/twin_q_out/BiasAdd*
N*
T0*+
_output_shapes
:џџџџџџџџџ*

axis 
h
&default_policy/Min_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 
Ў
default_policy/Min_1Mindefault_policy/Min_1/input&default_policy/Min_1/reduction_indices*
T0*

Tidx0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims( 

default_policy/SqueezeSqueeze3default_policy/model_1_1/sequential_1/q_out/BiasAdd*
T0*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims

Ђ
default_policy/Squeeze_1Squeeze8default_policy/model_2_1/sequential_2/twin_q_out/BiasAdd*
T0*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims


default_policy/mul_12Muldefault_policy/Expdefault_policy/ExpandDims_1*
T0*'
_output_shapes
:џџџџџџџџџ
{
default_policy/sub_28Subdefault_policy/Min_1default_policy/mul_12*
T0*'
_output_shapes
:џџџџџџџџџ

default_policy/Squeeze_2Squeezedefault_policy/sub_28*
T0*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims

~
default_policy/CastCastdefault_policy/dones*

DstT0*

SrcT0
*
Truncate( *#
_output_shapes
:џџџџџџџџџ
\
default_policy/sub_29/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
x
default_policy/sub_29Subdefault_policy/sub_29/xdefault_policy/Cast*
T0*#
_output_shapes
:џџџџџџџџџ
{
default_policy/mul_13Muldefault_policy/sub_29default_policy/Squeeze_2*
T0*#
_output_shapes
:џџџџџџџџџ
\
default_policy/mul_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?
z
default_policy/mul_14Muldefault_policy/mul_14/xdefault_policy/mul_13*
T0*#
_output_shapes
:џџџџџџџџџ
{
default_policy/add_17AddV2default_policy/rewardsdefault_policy/mul_14*
T0*#
_output_shapes
:џџџџџџџџџ
p
default_policy/StopGradientStopGradientdefault_policy/add_17*
T0*#
_output_shapes
:џџџџџџџџџ

default_policy/sub_30Subdefault_policy/Squeezedefault_policy/StopGradient*
T0*#
_output_shapes
:џџџџџџџџџ
^
default_policy/AbsAbsdefault_policy/sub_30*
T0*#
_output_shapes
:џџџџџџџџџ

default_policy/sub_31Subdefault_policy/Squeeze_1default_policy/StopGradient*
T0*#
_output_shapes
:џџџџџџџџџ
`
default_policy/Abs_1Absdefault_policy/sub_31*
T0*#
_output_shapes
:џџџџџџџџџ
v
default_policy/add_18AddV2default_policy/Absdefault_policy/Abs_1*
T0*#
_output_shapes
:џџџџџџџџџ
\
default_policy/mul_15/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
z
default_policy/mul_15Muldefault_policy/mul_15/xdefault_policy/add_18*
T0*#
_output_shapes
:џџџџџџџџџ

 default_policy/SquaredDifferenceSquaredDifferencedefault_policy/Squeezedefault_policy/StopGradient*
T0*#
_output_shapes
:џџџџџџџџџ
p
%default_policy/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
Ђ
default_policy/MeanMean default_policy/SquaredDifference%default_policy/Mean/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
\
default_policy/mul_16/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
k
default_policy/mul_16Muldefault_policy/mul_16/xdefault_policy/Mean*
T0*
_output_shapes
: 

"default_policy/SquaredDifference_1SquaredDifferencedefault_policy/Squeeze_1default_policy/StopGradient*
T0*#
_output_shapes
:џџџџџџџџџ
r
'default_policy/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
Ј
default_policy/Mean_1Mean"default_policy/SquaredDifference_1'default_policy/Mean_1/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
\
default_policy/mul_17/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
m
default_policy/mul_17Muldefault_policy/mul_17/xdefault_policy/Mean_1*
T0*
_output_shapes
: 
\
default_policy/add_19/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Р

default_policy/add_19AddV2default_policy/ExpandDimsdefault_policy/add_19/y*
T0*'
_output_shapes
:џџџџџџџџџ
v
default_policy/StopGradient_1StopGradientdefault_policy/add_19*
T0*'
_output_shapes
:џџџџџџџџџ
n
default_policy/ReadVariableOpReadVariableOpdefault_policy/log_alpha*
_output_shapes
: *
dtype0

default_policy/mul_18Muldefault_policy/ReadVariableOpdefault_policy/StopGradient_1*
T0*'
_output_shapes
:џџџџџџџџџ
e
default_policy/ConstConst*
_output_shapes
:*
dtype0*
valueB"       

default_policy/Mean_2Meandefault_policy/mul_18default_policy/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
Q
default_policy/NegNegdefault_policy/Mean_2*
T0*
_output_shapes
: 
}
default_policy/mul_19Muldefault_policy/Expdefault_policy/ExpandDims*
T0*'
_output_shapes
:џџџџџџџџџ
y
default_policy/sub_32Subdefault_policy/mul_19default_policy/Min*
T0*'
_output_shapes
:џџџџџџџџџ
g
default_policy/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       

default_policy/Mean_3Meandefault_policy/sub_32default_policy/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
s
default_policy/AddNAddNdefault_policy/mul_16default_policy/mul_17*
N*
T0*
_output_shapes
: 
k
default_policy/add_20AddV2default_policy/Mean_3default_policy/AddN*
T0*
_output_shapes
: 
j
default_policy/add_21AddV2default_policy/add_20default_policy/Neg*
T0*
_output_shapes
: 
`
default_policy/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 

default_policy/Mean_4Meandefault_policy/mul_15default_policy/Const_2*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
U
default_policy/RankConst*
_output_shapes
: *
dtype0*
value	B : 
\
default_policy/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
\
default_policy/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :

default_policy/rangeRangedefault_policy/range/startdefault_policy/Rankdefault_policy/range/delta*

Tidx0*
_output_shapes
: 

default_policy/Mean_5Meandefault_policy/Mean_3default_policy/range*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 

default_policy/Rank_1/packedPackdefault_policy/mul_16default_policy/mul_17*
N*
T0*
_output_shapes
:*

axis 
W
default_policy/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :
^
default_policy/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 
^
default_policy/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :

default_policy/range_1Rangedefault_policy/range_1/startdefault_policy/Rank_1default_policy/range_1/delta*

Tidx0*
_output_shapes
:

default_policy/Mean_6/inputPackdefault_policy/mul_16default_policy/mul_17*
N*
T0*
_output_shapes
:*

axis 

default_policy/Mean_6Meandefault_policy/Mean_6/inputdefault_policy/range_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
W
default_policy/Rank_2Const*
_output_shapes
: *
dtype0*
value	B : 
^
default_policy/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 
^
default_policy/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :

default_policy/range_2Rangedefault_policy/range_2/startdefault_policy/Rank_2default_policy/range_2/delta*

Tidx0*
_output_shapes
: 

default_policy/Mean_7Meandefault_policy/Negdefault_policy/range_2*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
W
default_policy/Rank_3Const*
_output_shapes
: *
dtype0*
value	B : 
^
default_policy/range_3/startConst*
_output_shapes
: *
dtype0*
value	B : 
^
default_policy/range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :

default_policy/range_3Rangedefault_policy/range_3/startdefault_policy/Rank_3default_policy/range_3/delta*

Tidx0*
_output_shapes
: 

default_policy/Mean_8Meandefault_policy/Expdefault_policy/range_3*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
a
default_policy/Const_3Const*
_output_shapes
: *
dtype0	*
valueB	 R
ўџџџџџџџџ
g
default_policy/Const_4Const*
_output_shapes
:*
dtype0*
valueB"       
Ј
default_policy/Mean_9Mean3default_policy/model_1_1/sequential_1/q_out/BiasAdddefault_policy/Const_4*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
g
default_policy/Const_5Const*
_output_shapes
:*
dtype0*
valueB"       
Є
default_policy/MaxMax3default_policy/model_1_1/sequential_1/q_out/BiasAdddefault_policy/Const_5*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
g
default_policy/Const_6Const*
_output_shapes
:*
dtype0*
valueB"       
І
default_policy/Min_2Min3default_policy/model_1_1/sequential_1/q_out/BiasAdddefault_policy/Const_6*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
a
default_policy/gradients/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
m
(default_policy/gradients/grad_ys_0/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
Ї
"default_policy/gradients/grad_ys_0Filldefault_policy/gradients/Shape(default_policy/gradients/grad_ys_0/Const*
T0*
_output_shapes
: *

index_type0

Adefault_policy/gradients/default_policy/Mean_3_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
ф
;default_policy/gradients/default_policy/Mean_3_grad/ReshapeReshape"default_policy/gradients/grad_ys_0Adefault_policy/gradients/default_policy/Mean_3_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:

9default_policy/gradients/default_policy/Mean_3_grad/ShapeShapedefault_policy/sub_32*
T0*
_output_shapes
:*
out_type0
ќ
8default_policy/gradients/default_policy/Mean_3_grad/TileTile;default_policy/gradients/default_policy/Mean_3_grad/Reshape9default_policy/gradients/default_policy/Mean_3_grad/Shape*
T0*

Tmultiples0*'
_output_shapes
:џџџџџџџџџ

;default_policy/gradients/default_policy/Mean_3_grad/Shape_1Shapedefault_policy/sub_32*
T0*
_output_shapes
:*
out_type0
~
;default_policy/gradients/default_policy/Mean_3_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 

9default_policy/gradients/default_policy/Mean_3_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
і
8default_policy/gradients/default_policy/Mean_3_grad/ProdProd;default_policy/gradients/default_policy/Mean_3_grad/Shape_19default_policy/gradients/default_policy/Mean_3_grad/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 

;default_policy/gradients/default_policy/Mean_3_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
њ
:default_policy/gradients/default_policy/Mean_3_grad/Prod_1Prod;default_policy/gradients/default_policy/Mean_3_grad/Shape_2;default_policy/gradients/default_policy/Mean_3_grad/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 

=default_policy/gradients/default_policy/Mean_3_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :
т
;default_policy/gradients/default_policy/Mean_3_grad/MaximumMaximum:default_policy/gradients/default_policy/Mean_3_grad/Prod_1=default_policy/gradients/default_policy/Mean_3_grad/Maximum/y*
T0*
_output_shapes
: 
р
<default_policy/gradients/default_policy/Mean_3_grad/floordivFloorDiv8default_policy/gradients/default_policy/Mean_3_grad/Prod;default_policy/gradients/default_policy/Mean_3_grad/Maximum*
T0*
_output_shapes
: 
О
8default_policy/gradients/default_policy/Mean_3_grad/CastCast<default_policy/gradients/default_policy/Mean_3_grad/floordiv*

DstT0*

SrcT0*
Truncate( *
_output_shapes
: 
ь
;default_policy/gradients/default_policy/Mean_3_grad/truedivRealDiv8default_policy/gradients/default_policy/Mean_3_grad/Tile8default_policy/gradients/default_policy/Mean_3_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ

9default_policy/gradients/default_policy/sub_32_grad/ShapeShapedefault_policy/mul_19*
T0*
_output_shapes
:*
out_type0

;default_policy/gradients/default_policy/sub_32_grad/Shape_1Shapedefault_policy/Min*
T0*
_output_shapes
:*
out_type0

Idefault_policy/gradients/default_policy/sub_32_grad/BroadcastGradientArgsBroadcastGradientArgs9default_policy/gradients/default_policy/sub_32_grad/Shape;default_policy/gradients/default_policy/sub_32_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

7default_policy/gradients/default_policy/sub_32_grad/SumSum;default_policy/gradients/default_policy/Mean_3_grad/truedivIdefault_policy/gradients/default_policy/sub_32_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Ђ
;default_policy/gradients/default_policy/sub_32_grad/ReshapeReshape7default_policy/gradients/default_policy/sub_32_grad/Sum9default_policy/gradients/default_policy/sub_32_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
­
7default_policy/gradients/default_policy/sub_32_grad/NegNeg;default_policy/gradients/default_policy/Mean_3_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ

9default_policy/gradients/default_policy/sub_32_grad/Sum_1Sum7default_policy/gradients/default_policy/sub_32_grad/NegKdefault_policy/gradients/default_policy/sub_32_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Ј
=default_policy/gradients/default_policy/sub_32_grad/Reshape_1Reshape9default_policy/gradients/default_policy/sub_32_grad/Sum_1;default_policy/gradients/default_policy/sub_32_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
ђ
Ddefault_policy/gradients/default_policy/sub_32_grad/tuple/group_depsNoOp<^default_policy/gradients/default_policy/sub_32_grad/Reshape>^default_policy/gradients/default_policy/sub_32_grad/Reshape_1*&
 _has_manual_control_dependencies(
о
Ldefault_policy/gradients/default_policy/sub_32_grad/tuple/control_dependencyIdentity;default_policy/gradients/default_policy/sub_32_grad/ReshapeE^default_policy/gradients/default_policy/sub_32_grad/tuple/group_deps*
T0*N
_classD
B@loc:@default_policy/gradients/default_policy/sub_32_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
ф
Ndefault_policy/gradients/default_policy/sub_32_grad/tuple/control_dependency_1Identity=default_policy/gradients/default_policy/sub_32_grad/Reshape_1E^default_policy/gradients/default_policy/sub_32_grad/tuple/group_deps*
T0*P
_classF
DBloc:@default_policy/gradients/default_policy/sub_32_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

9default_policy/gradients/default_policy/mul_19_grad/ShapeShapedefault_policy/Exp*
T0*
_output_shapes
: *
out_type0

;default_policy/gradients/default_policy/mul_19_grad/Shape_1Shapedefault_policy/ExpandDims*
T0*
_output_shapes
:*
out_type0

Idefault_policy/gradients/default_policy/mul_19_grad/BroadcastGradientArgsBroadcastGradientArgs9default_policy/gradients/default_policy/mul_19_grad/Shape;default_policy/gradients/default_policy/mul_19_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
й
7default_policy/gradients/default_policy/mul_19_grad/MulMulLdefault_policy/gradients/default_policy/sub_32_grad/tuple/control_dependencydefault_policy/ExpandDims*
T0*'
_output_shapes
:џџџџџџџџџ

7default_policy/gradients/default_policy/mul_19_grad/SumSum7default_policy/gradients/default_policy/mul_19_grad/MulIdefault_policy/gradients/default_policy/mul_19_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

;default_policy/gradients/default_policy/mul_19_grad/ReshapeReshape7default_policy/gradients/default_policy/mul_19_grad/Sum9default_policy/gradients/default_policy/mul_19_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*
_output_shapes
: 
д
9default_policy/gradients/default_policy/mul_19_grad/Mul_1Muldefault_policy/ExpLdefault_policy/gradients/default_policy/sub_32_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ

9default_policy/gradients/default_policy/mul_19_grad/Sum_1Sum9default_policy/gradients/default_policy/mul_19_grad/Mul_1Kdefault_policy/gradients/default_policy/mul_19_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Ј
=default_policy/gradients/default_policy/mul_19_grad/Reshape_1Reshape9default_policy/gradients/default_policy/mul_19_grad/Sum_1;default_policy/gradients/default_policy/mul_19_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
ђ
Ddefault_policy/gradients/default_policy/mul_19_grad/tuple/group_depsNoOp<^default_policy/gradients/default_policy/mul_19_grad/Reshape>^default_policy/gradients/default_policy/mul_19_grad/Reshape_1*&
 _has_manual_control_dependencies(
Э
Ldefault_policy/gradients/default_policy/mul_19_grad/tuple/control_dependencyIdentity;default_policy/gradients/default_policy/mul_19_grad/ReshapeE^default_policy/gradients/default_policy/mul_19_grad/tuple/group_deps*
T0*N
_classD
B@loc:@default_policy/gradients/default_policy/mul_19_grad/Reshape*
_output_shapes
: 
ф
Ndefault_policy/gradients/default_policy/mul_19_grad/tuple/control_dependency_1Identity=default_policy/gradients/default_policy/mul_19_grad/Reshape_1E^default_policy/gradients/default_policy/mul_19_grad/tuple/group_deps*
T0*P
_classF
DBloc:@default_policy/gradients/default_policy/mul_19_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

6default_policy/gradients/default_policy/Min_grad/ShapeShapedefault_policy/Min/input*
T0*
_output_shapes
:*
out_type0
w
5default_policy/gradients/default_policy/Min_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
Л
4default_policy/gradients/default_policy/Min_grad/addAddV2$default_policy/Min/reduction_indices5default_policy/gradients/default_policy/Min_grad/Size*
T0*
_output_shapes
: 
Ю
4default_policy/gradients/default_policy/Min_grad/modFloorMod4default_policy/gradients/default_policy/Min_grad/add5default_policy/gradients/default_policy/Min_grad/Size*
T0*
_output_shapes
: 
{
8default_policy/gradients/default_policy/Min_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
~
<default_policy/gradients/default_policy/Min_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
~
<default_policy/gradients/default_policy/Min_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :

6default_policy/gradients/default_policy/Min_grad/rangeRange<default_policy/gradients/default_policy/Min_grad/range/start5default_policy/gradients/default_policy/Min_grad/Size<default_policy/gradients/default_policy/Min_grad/range/delta*

Tidx0*
_output_shapes
:
}
;default_policy/gradients/default_policy/Min_grad/ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :
ч
5default_policy/gradients/default_policy/Min_grad/onesFill8default_policy/gradients/default_policy/Min_grad/Shape_1;default_policy/gradients/default_policy/Min_grad/ones/Const*
T0*
_output_shapes
: *

index_type0
к
>default_policy/gradients/default_policy/Min_grad/DynamicStitchDynamicStitch6default_policy/gradients/default_policy/Min_grad/range4default_policy/gradients/default_policy/Min_grad/mod6default_policy/gradients/default_policy/Min_grad/Shape5default_policy/gradients/default_policy/Min_grad/ones*
N*
T0*
_output_shapes
:
э
8default_policy/gradients/default_policy/Min_grad/ReshapeReshapedefault_policy/Min>default_policy/gradients/default_policy/Min_grad/DynamicStitch*
T0*
Tshape0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ћ
:default_policy/gradients/default_policy/Min_grad/Reshape_1ReshapeNdefault_policy/gradients/default_policy/sub_32_grad/tuple/control_dependency_1>default_policy/gradients/default_policy/Min_grad/DynamicStitch*
T0*
Tshape0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
ђ
6default_policy/gradients/default_policy/Min_grad/EqualEqual8default_policy/gradients/default_policy/Min_grad/Reshapedefault_policy/Min/input*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
incompatible_shape_error(
г
5default_policy/gradients/default_policy/Min_grad/CastCast6default_policy/gradients/default_policy/Min_grad/Equal*

DstT0*

SrcT0
*
Truncate( *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
№
4default_policy/gradients/default_policy/Min_grad/SumSum5default_policy/gradients/default_policy/Min_grad/Cast$default_policy/Min/reduction_indices*
T0*

Tidx0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims( 

:default_policy/gradients/default_policy/Min_grad/Reshape_2Reshape4default_policy/gradients/default_policy/Min_grad/Sum>default_policy/gradients/default_policy/Min_grad/DynamicStitch*
T0*
Tshape0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
ѕ
8default_policy/gradients/default_policy/Min_grad/truedivRealDiv5default_policy/gradients/default_policy/Min_grad/Cast:default_policy/gradients/default_policy/Min_grad/Reshape_2*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
ч
4default_policy/gradients/default_policy/Min_grad/mulMul8default_policy/gradients/default_policy/Min_grad/truediv:default_policy/gradients/default_policy/Min_grad/Reshape_1*
T0*+
_output_shapes
:џџџџџџџџџ

=default_policy/gradients/default_policy/ExpandDims_grad/ShapeShapedefault_policy/sub_19*
T0*
_output_shapes
:*
out_type0

?default_policy/gradients/default_policy/ExpandDims_grad/ReshapeReshapeNdefault_policy/gradients/default_policy/mul_19_grad/tuple/control_dependency_1=default_policy/gradients/default_policy/ExpandDims_grad/Shape*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ

>default_policy/gradients/default_policy/Min/input_grad/unstackUnpack4default_policy/gradients/default_policy/Min_grad/mul*
T0*&
 _has_manual_control_dependencies(*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*

axis *	
num
И
Gdefault_policy/gradients/default_policy/Min/input_grad/tuple/group_depsNoOp?^default_policy/gradients/default_policy/Min/input_grad/unstack*&
 _has_manual_control_dependencies(

Odefault_policy/gradients/default_policy/Min/input_grad/tuple/control_dependencyIdentity>default_policy/gradients/default_policy/Min/input_grad/unstackH^default_policy/gradients/default_policy/Min/input_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@default_policy/gradients/default_policy/Min/input_grad/unstack*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ

Qdefault_policy/gradients/default_policy/Min/input_grad/tuple/control_dependency_1Identity@default_policy/gradients/default_policy/Min/input_grad/unstack:1H^default_policy/gradients/default_policy/Min/input_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@default_policy/gradients/default_policy/Min/input_grad/unstack*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ

9default_policy/gradients/default_policy/sub_19_grad/ShapeShapedefault_policy/Sum_8*
T0*
_output_shapes
:*
out_type0

;default_policy/gradients/default_policy/sub_19_grad/Shape_1Shapedefault_policy/Sum_9*
T0*
_output_shapes
:*
out_type0

Idefault_policy/gradients/default_policy/sub_19_grad/BroadcastGradientArgsBroadcastGradientArgs9default_policy/gradients/default_policy/sub_19_grad/Shape;default_policy/gradients/default_policy/sub_19_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

7default_policy/gradients/default_policy/sub_19_grad/SumSum?default_policy/gradients/default_policy/ExpandDims_grad/ReshapeIdefault_policy/gradients/default_policy/sub_19_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

;default_policy/gradients/default_policy/sub_19_grad/ReshapeReshape7default_policy/gradients/default_policy/sub_19_grad/Sum9default_policy/gradients/default_policy/sub_19_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*#
_output_shapes
:џџџџџџџџџ
­
7default_policy/gradients/default_policy/sub_19_grad/NegNeg?default_policy/gradients/default_policy/ExpandDims_grad/Reshape*
T0*#
_output_shapes
:џџџџџџџџџ

9default_policy/gradients/default_policy/sub_19_grad/Sum_1Sum7default_policy/gradients/default_policy/sub_19_grad/NegKdefault_policy/gradients/default_policy/sub_19_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Є
=default_policy/gradients/default_policy/sub_19_grad/Reshape_1Reshape9default_policy/gradients/default_policy/sub_19_grad/Sum_1;default_policy/gradients/default_policy/sub_19_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*#
_output_shapes
:џџџџџџџџџ
ђ
Ddefault_policy/gradients/default_policy/sub_19_grad/tuple/group_depsNoOp<^default_policy/gradients/default_policy/sub_19_grad/Reshape>^default_policy/gradients/default_policy/sub_19_grad/Reshape_1*&
 _has_manual_control_dependencies(
к
Ldefault_policy/gradients/default_policy/sub_19_grad/tuple/control_dependencyIdentity;default_policy/gradients/default_policy/sub_19_grad/ReshapeE^default_policy/gradients/default_policy/sub_19_grad/tuple/group_deps*
T0*N
_classD
B@loc:@default_policy/gradients/default_policy/sub_19_grad/Reshape*#
_output_shapes
:џџџџџџџџџ
р
Ndefault_policy/gradients/default_policy/sub_19_grad/tuple/control_dependency_1Identity=default_policy/gradients/default_policy/sub_19_grad/Reshape_1E^default_policy/gradients/default_policy/sub_19_grad/tuple/group_deps*
T0*P
_classF
DBloc:@default_policy/gradients/default_policy/sub_19_grad/Reshape_1*#
_output_shapes
:џџџџџџџџџ
Ё
]default_policy/gradients/default_policy/model_1_2/sequential_1/q_out/BiasAdd_grad/BiasAddGradBiasAddGradOdefault_policy/gradients/default_policy/Min/input_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:*
data_formatNHWC
Ф
bdefault_policy/gradients/default_policy/model_1_2/sequential_1/q_out/BiasAdd_grad/tuple/group_depsNoOpP^default_policy/gradients/default_policy/Min/input_grad/tuple/control_dependency^^default_policy/gradients/default_policy/model_1_2/sequential_1/q_out/BiasAdd_grad/BiasAddGrad*&
 _has_manual_control_dependencies(
Б
jdefault_policy/gradients/default_policy/model_1_2/sequential_1/q_out/BiasAdd_grad/tuple/control_dependencyIdentityOdefault_policy/gradients/default_policy/Min/input_grad/tuple/control_dependencyc^default_policy/gradients/default_policy/model_1_2/sequential_1/q_out/BiasAdd_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@default_policy/gradients/default_policy/Min/input_grad/unstack*'
_output_shapes
:џџџџџџџџџ
г
ldefault_policy/gradients/default_policy/model_1_2/sequential_1/q_out/BiasAdd_grad/tuple/control_dependency_1Identity]default_policy/gradients/default_policy/model_1_2/sequential_1/q_out/BiasAdd_grad/BiasAddGradc^default_policy/gradients/default_policy/model_1_2/sequential_1/q_out/BiasAdd_grad/tuple/group_deps*
T0*p
_classf
dbloc:@default_policy/gradients/default_policy/model_1_2/sequential_1/q_out/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
Ј
bdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_out/BiasAdd_grad/BiasAddGradBiasAddGradQdefault_policy/gradients/default_policy/Min/input_grad/tuple/control_dependency_1*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:*
data_formatNHWC
а
gdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_out/BiasAdd_grad/tuple/group_depsNoOpR^default_policy/gradients/default_policy/Min/input_grad/tuple/control_dependency_1c^default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_out/BiasAdd_grad/BiasAddGrad*&
 _has_manual_control_dependencies(
Н
odefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_out/BiasAdd_grad/tuple/control_dependencyIdentityQdefault_policy/gradients/default_policy/Min/input_grad/tuple/control_dependency_1h^default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_out/BiasAdd_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@default_policy/gradients/default_policy/Min/input_grad/unstack*'
_output_shapes
:џџџџџџџџџ
ч
qdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_out/BiasAdd_grad/tuple/control_dependency_1Identitybdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_out/BiasAdd_grad/BiasAddGradh^default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_out/BiasAdd_grad/tuple/group_deps*
T0*u
_classk
igloc:@default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_out/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

8default_policy/gradients/default_policy/Sum_8_grad/ShapeShapedefault_policy/clip_by_value_16*
T0*
_output_shapes
:*
out_type0
Ц
7default_policy/gradients/default_policy/Sum_8_grad/SizeConst*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_8_grad/Shape*
_output_shapes
: *
dtype0*
value	B :

6default_policy/gradients/default_policy/Sum_8_grad/addAddV2&default_policy/Sum_8/reduction_indices7default_policy/gradients/default_policy/Sum_8_grad/Size*
T0*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_8_grad/Shape*
_output_shapes
: 
Ё
6default_policy/gradients/default_policy/Sum_8_grad/modFloorMod6default_policy/gradients/default_policy/Sum_8_grad/add7default_policy/gradients/default_policy/Sum_8_grad/Size*
T0*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_8_grad/Shape*
_output_shapes
: 
Ъ
:default_policy/gradients/default_policy/Sum_8_grad/Shape_1Const*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_8_grad/Shape*
_output_shapes
: *
dtype0*
valueB 
Э
>default_policy/gradients/default_policy/Sum_8_grad/range/startConst*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_8_grad/Shape*
_output_shapes
: *
dtype0*
value	B : 
Э
>default_policy/gradients/default_policy/Sum_8_grad/range/deltaConst*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_8_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
я
8default_policy/gradients/default_policy/Sum_8_grad/rangeRange>default_policy/gradients/default_policy/Sum_8_grad/range/start7default_policy/gradients/default_policy/Sum_8_grad/Size>default_policy/gradients/default_policy/Sum_8_grad/range/delta*

Tidx0*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_8_grad/Shape*
_output_shapes
:
Ь
=default_policy/gradients/default_policy/Sum_8_grad/ones/ConstConst*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_8_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
К
7default_policy/gradients/default_policy/Sum_8_grad/onesFill:default_policy/gradients/default_policy/Sum_8_grad/Shape_1=default_policy/gradients/default_policy/Sum_8_grad/ones/Const*
T0*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_8_grad/Shape*
_output_shapes
: *

index_type0
Б
@default_policy/gradients/default_policy/Sum_8_grad/DynamicStitchDynamicStitch8default_policy/gradients/default_policy/Sum_8_grad/range6default_policy/gradients/default_policy/Sum_8_grad/mod8default_policy/gradients/default_policy/Sum_8_grad/Shape7default_policy/gradients/default_policy/Sum_8_grad/ones*
N*
T0*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_8_grad/Shape*
_output_shapes
:

:default_policy/gradients/default_policy/Sum_8_grad/ReshapeReshapeLdefault_policy/gradients/default_policy/sub_19_grad/tuple/control_dependency@default_policy/gradients/default_policy/Sum_8_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

>default_policy/gradients/default_policy/Sum_8_grad/BroadcastToBroadcastTo:default_policy/gradients/default_policy/Sum_8_grad/Reshape8default_policy/gradients/default_policy/Sum_8_grad/Shape*
T0*

Tidx0*'
_output_shapes
:џџџџџџџџџ

8default_policy/gradients/default_policy/Sum_9_grad/ShapeShapedefault_policy/Log_4*
T0*
_output_shapes
:*
out_type0
Ц
7default_policy/gradients/default_policy/Sum_9_grad/SizeConst*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_9_grad/Shape*
_output_shapes
: *
dtype0*
value	B :

6default_policy/gradients/default_policy/Sum_9_grad/addAddV2&default_policy/Sum_9/reduction_indices7default_policy/gradients/default_policy/Sum_9_grad/Size*
T0*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_9_grad/Shape*
_output_shapes
: 
Ё
6default_policy/gradients/default_policy/Sum_9_grad/modFloorMod6default_policy/gradients/default_policy/Sum_9_grad/add7default_policy/gradients/default_policy/Sum_9_grad/Size*
T0*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_9_grad/Shape*
_output_shapes
: 
Ъ
:default_policy/gradients/default_policy/Sum_9_grad/Shape_1Const*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_9_grad/Shape*
_output_shapes
: *
dtype0*
valueB 
Э
>default_policy/gradients/default_policy/Sum_9_grad/range/startConst*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_9_grad/Shape*
_output_shapes
: *
dtype0*
value	B : 
Э
>default_policy/gradients/default_policy/Sum_9_grad/range/deltaConst*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_9_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
я
8default_policy/gradients/default_policy/Sum_9_grad/rangeRange>default_policy/gradients/default_policy/Sum_9_grad/range/start7default_policy/gradients/default_policy/Sum_9_grad/Size>default_policy/gradients/default_policy/Sum_9_grad/range/delta*

Tidx0*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_9_grad/Shape*
_output_shapes
:
Ь
=default_policy/gradients/default_policy/Sum_9_grad/ones/ConstConst*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_9_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
К
7default_policy/gradients/default_policy/Sum_9_grad/onesFill:default_policy/gradients/default_policy/Sum_9_grad/Shape_1=default_policy/gradients/default_policy/Sum_9_grad/ones/Const*
T0*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_9_grad/Shape*
_output_shapes
: *

index_type0
Б
@default_policy/gradients/default_policy/Sum_9_grad/DynamicStitchDynamicStitch8default_policy/gradients/default_policy/Sum_9_grad/range6default_policy/gradients/default_policy/Sum_9_grad/mod8default_policy/gradients/default_policy/Sum_9_grad/Shape7default_policy/gradients/default_policy/Sum_9_grad/ones*
N*
T0*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_9_grad/Shape*
_output_shapes
:
 
:default_policy/gradients/default_policy/Sum_9_grad/ReshapeReshapeNdefault_policy/gradients/default_policy/sub_19_grad/tuple/control_dependency_1@default_policy/gradients/default_policy/Sum_9_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Љ
>default_policy/gradients/default_policy/Sum_9_grad/BroadcastToBroadcastTo:default_policy/gradients/default_policy/Sum_9_grad/Reshape8default_policy/gradients/default_policy/Sum_9_grad/Shape*
T0*

Tidx0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ

Wdefault_policy/gradients/default_policy/model_1_2/sequential_1/q_out/MatMul_grad/MatMulMatMuljdefault_policy/gradients/default_policy/model_1_2/sequential_1/q_out/BiasAdd_grad/tuple/control_dependencyAdefault_policy/model_1_2/sequential_1/q_out/MatMul/ReadVariableOp*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
ў
Ydefault_policy/gradients/default_policy/model_1_2/sequential_1/q_out/MatMul_grad/MatMul_1MatMul5default_policy/model_1_2/sequential_1/q_hidden_1/Relujdefault_policy/gradients/default_policy/model_1_2/sequential_1/q_out/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:	*
transpose_a(*
transpose_b( 
Ч
adefault_policy/gradients/default_policy/model_1_2/sequential_1/q_out/MatMul_grad/tuple/group_depsNoOpX^default_policy/gradients/default_policy/model_1_2/sequential_1/q_out/MatMul_grad/MatMulZ^default_policy/gradients/default_policy/model_1_2/sequential_1/q_out/MatMul_grad/MatMul_1*&
 _has_manual_control_dependencies(
б
idefault_policy/gradients/default_policy/model_1_2/sequential_1/q_out/MatMul_grad/tuple/control_dependencyIdentityWdefault_policy/gradients/default_policy/model_1_2/sequential_1/q_out/MatMul_grad/MatMulb^default_policy/gradients/default_policy/model_1_2/sequential_1/q_out/MatMul_grad/tuple/group_deps*
T0*j
_class`
^\loc:@default_policy/gradients/default_policy/model_1_2/sequential_1/q_out/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
Ю
kdefault_policy/gradients/default_policy/model_1_2/sequential_1/q_out/MatMul_grad/tuple/control_dependency_1IdentityYdefault_policy/gradients/default_policy/model_1_2/sequential_1/q_out/MatMul_grad/MatMul_1b^default_policy/gradients/default_policy/model_1_2/sequential_1/q_out/MatMul_grad/tuple/group_deps*
T0*l
_classb
`^loc:@default_policy/gradients/default_policy/model_1_2/sequential_1/q_out/MatMul_grad/MatMul_1*
_output_shapes
:	
 
\default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_out/MatMul_grad/MatMulMatMulodefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_out/BiasAdd_grad/tuple/control_dependencyFdefault_policy/model_2_2/sequential_2/twin_q_out/MatMul/ReadVariableOp*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(

^default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_out/MatMul_grad/MatMul_1MatMul:default_policy/model_2_2/sequential_2/twin_q_hidden_1/Reluodefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_out/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:	*
transpose_a(*
transpose_b( 
ж
fdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_out/MatMul_grad/tuple/group_depsNoOp]^default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_out/MatMul_grad/MatMul_^default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_out/MatMul_grad/MatMul_1*&
 _has_manual_control_dependencies(
х
ndefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_out/MatMul_grad/tuple/control_dependencyIdentity\default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_out/MatMul_grad/MatMulg^default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_out/MatMul_grad/tuple/group_deps*
T0*o
_classe
caloc:@default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_out/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
т
pdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_out/MatMul_grad/tuple/control_dependency_1Identity^default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_out/MatMul_grad/MatMul_1g^default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_out/MatMul_grad/tuple/group_deps*
T0*q
_classg
ecloc:@default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_out/MatMul_grad/MatMul_1*
_output_shapes
:	
Њ
Cdefault_policy/gradients/default_policy/clip_by_value_16_grad/ShapeShape'default_policy/clip_by_value_16/Minimum*
T0*
_output_shapes
:*
out_type0

Edefault_policy/gradients/default_policy/clip_by_value_16_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
Ч
Hdefault_policy/gradients/default_policy/clip_by_value_16_grad/zeros_like	ZerosLike>default_policy/gradients/default_policy/Sum_8_grad/BroadcastTo*
T0*'
_output_shapes
:џџџџџџџџџ
и
Jdefault_policy/gradients/default_policy/clip_by_value_16_grad/GreaterEqualGreaterEqual'default_policy/clip_by_value_16/Minimum!default_policy/clip_by_value_16/y*
T0*'
_output_shapes
:џџџџџџџџџ
Е
Sdefault_policy/gradients/default_policy/clip_by_value_16_grad/BroadcastGradientArgsBroadcastGradientArgsCdefault_policy/gradients/default_policy/clip_by_value_16_grad/ShapeEdefault_policy/gradients/default_policy/clip_by_value_16_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
к
Fdefault_policy/gradients/default_policy/clip_by_value_16_grad/SelectV2SelectV2Jdefault_policy/gradients/default_policy/clip_by_value_16_grad/GreaterEqual>default_policy/gradients/default_policy/Sum_8_grad/BroadcastToHdefault_policy/gradients/default_policy/clip_by_value_16_grad/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
Ѕ
Adefault_policy/gradients/default_policy/clip_by_value_16_grad/SumSumFdefault_policy/gradients/default_policy/clip_by_value_16_grad/SelectV2Sdefault_policy/gradients/default_policy/clip_by_value_16_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Р
Edefault_policy/gradients/default_policy/clip_by_value_16_grad/ReshapeReshapeAdefault_policy/gradients/default_policy/clip_by_value_16_grad/SumCdefault_policy/gradients/default_policy/clip_by_value_16_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
м
Hdefault_policy/gradients/default_policy/clip_by_value_16_grad/SelectV2_1SelectV2Jdefault_policy/gradients/default_policy/clip_by_value_16_grad/GreaterEqualHdefault_policy/gradients/default_policy/clip_by_value_16_grad/zeros_like>default_policy/gradients/default_policy/Sum_8_grad/BroadcastTo*
T0*'
_output_shapes
:џџџџџџџџџ
Ћ
Cdefault_policy/gradients/default_policy/clip_by_value_16_grad/Sum_1SumHdefault_policy/gradients/default_policy/clip_by_value_16_grad/SelectV2_1Udefault_policy/gradients/default_policy/clip_by_value_16_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Е
Gdefault_policy/gradients/default_policy/clip_by_value_16_grad/Reshape_1ReshapeCdefault_policy/gradients/default_policy/clip_by_value_16_grad/Sum_1Edefault_policy/gradients/default_policy/clip_by_value_16_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*
_output_shapes
: 

Ndefault_policy/gradients/default_policy/clip_by_value_16_grad/tuple/group_depsNoOpF^default_policy/gradients/default_policy/clip_by_value_16_grad/ReshapeH^default_policy/gradients/default_policy/clip_by_value_16_grad/Reshape_1*&
 _has_manual_control_dependencies(

Vdefault_policy/gradients/default_policy/clip_by_value_16_grad/tuple/control_dependencyIdentityEdefault_policy/gradients/default_policy/clip_by_value_16_grad/ReshapeO^default_policy/gradients/default_policy/clip_by_value_16_grad/tuple/group_deps*
T0*X
_classN
LJloc:@default_policy/gradients/default_policy/clip_by_value_16_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
ћ
Xdefault_policy/gradients/default_policy/clip_by_value_16_grad/tuple/control_dependency_1IdentityGdefault_policy/gradients/default_policy/clip_by_value_16_grad/Reshape_1O^default_policy/gradients/default_policy/clip_by_value_16_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@default_policy/gradients/default_policy/clip_by_value_16_grad/Reshape_1*
_output_shapes
: 
е
=default_policy/gradients/default_policy/Log_4_grad/Reciprocal
Reciprocaldefault_policy/add_12?^default_policy/gradients/default_policy/Sum_9_grad/BroadcastTo*
T0*'
_output_shapes
:џџџџџџџџџ
ю
6default_policy/gradients/default_policy/Log_4_grad/mulMul>default_policy/gradients/default_policy/Sum_9_grad/BroadcastTo=default_policy/gradients/default_policy/Log_4_grad/Reciprocal*
T0*'
_output_shapes
:џџџџџџџџџ
х
\default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/Relu_grad/ReluGradReluGradidefault_policy/gradients/default_policy/model_1_2/sequential_1/q_out/MatMul_grad/tuple/control_dependency5default_policy/model_1_2/sequential_1/q_hidden_1/Relu*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:џџџџџџџџџ
є
adefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/Relu_grad/ReluGradReluGradndefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_out/MatMul_grad/tuple/control_dependency:default_policy/model_2_2/sequential_2/twin_q_hidden_1/Relu*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:џџџџџџџџџ
Р
Kdefault_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/ShapeShape5default_policy/default_policy_Normal_2_3/log_prob/sub*
T0*
_output_shapes
:*
out_type0

Mdefault_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
ч
Pdefault_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/zeros_like	ZerosLikeVdefault_policy/gradients/default_policy/clip_by_value_16_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
№
Odefault_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/LessEqual	LessEqual5default_policy/default_policy_Normal_2_3/log_prob/sub)default_policy/clip_by_value_16/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
Э
[default_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsKdefault_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/ShapeMdefault_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Ndefault_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/SelectV2SelectV2Odefault_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/LessEqualVdefault_policy/gradients/default_policy/clip_by_value_16_grad/tuple/control_dependencyPdefault_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
Н
Idefault_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/SumSumNdefault_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/SelectV2[default_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
и
Mdefault_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/ReshapeReshapeIdefault_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/SumKdefault_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ

Pdefault_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/SelectV2_1SelectV2Odefault_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/LessEqualPdefault_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/zeros_likeVdefault_policy/gradients/default_policy/clip_by_value_16_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
У
Kdefault_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/Sum_1SumPdefault_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/SelectV2_1]default_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Э
Odefault_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/Reshape_1ReshapeKdefault_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/Sum_1Mdefault_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*
_output_shapes
: 
Ј
Vdefault_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/tuple/group_depsNoOpN^default_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/ReshapeP^default_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/Reshape_1*&
 _has_manual_control_dependencies(
І
^default_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/tuple/control_dependencyIdentityMdefault_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/ReshapeW^default_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/tuple/group_deps*
T0*`
_classV
TRloc:@default_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

`default_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/tuple/control_dependency_1IdentityOdefault_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/Reshape_1W^default_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/tuple/group_deps*
T0*b
_classX
VTloc:@default_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/Reshape_1*
_output_shapes
: 

9default_policy/gradients/default_policy/add_12_grad/ShapeShapedefault_policy/sub_18*
T0*
_output_shapes
:*
out_type0

;default_policy/gradients/default_policy/add_12_grad/Shape_1Shapedefault_policy/add_12/y*
T0*
_output_shapes
: *
out_type0

Idefault_policy/gradients/default_policy/add_12_grad/BroadcastGradientArgsBroadcastGradientArgs9default_policy/gradients/default_policy/add_12_grad/Shape;default_policy/gradients/default_policy/add_12_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

7default_policy/gradients/default_policy/add_12_grad/SumSum6default_policy/gradients/default_policy/Log_4_grad/mulIdefault_policy/gradients/default_policy/add_12_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Ђ
;default_policy/gradients/default_policy/add_12_grad/ReshapeReshape7default_policy/gradients/default_policy/add_12_grad/Sum9default_policy/gradients/default_policy/add_12_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ

9default_policy/gradients/default_policy/add_12_grad/Sum_1Sum6default_policy/gradients/default_policy/Log_4_grad/mulKdefault_policy/gradients/default_policy/add_12_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

=default_policy/gradients/default_policy/add_12_grad/Reshape_1Reshape9default_policy/gradients/default_policy/add_12_grad/Sum_1;default_policy/gradients/default_policy/add_12_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*
_output_shapes
: 
ђ
Ddefault_policy/gradients/default_policy/add_12_grad/tuple/group_depsNoOp<^default_policy/gradients/default_policy/add_12_grad/Reshape>^default_policy/gradients/default_policy/add_12_grad/Reshape_1*&
 _has_manual_control_dependencies(
о
Ldefault_policy/gradients/default_policy/add_12_grad/tuple/control_dependencyIdentity;default_policy/gradients/default_policy/add_12_grad/ReshapeE^default_policy/gradients/default_policy/add_12_grad/tuple/group_deps*
T0*N
_classD
B@loc:@default_policy/gradients/default_policy/add_12_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
г
Ndefault_policy/gradients/default_policy/add_12_grad/tuple/control_dependency_1Identity=default_policy/gradients/default_policy/add_12_grad/Reshape_1E^default_policy/gradients/default_policy/add_12_grad/tuple/group_deps*
T0*P
_classF
DBloc:@default_policy/gradients/default_policy/add_12_grad/Reshape_1*
_output_shapes
: 
Д
bdefault_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/BiasAdd_grad/BiasAddGradBiasAddGrad\default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/Relu_grad/ReluGrad*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:*
data_formatNHWC
л
gdefault_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/BiasAdd_grad/tuple/group_depsNoOpc^default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/BiasAdd_grad/BiasAddGrad]^default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/Relu_grad/ReluGrad*&
 _has_manual_control_dependencies(
ч
odefault_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/BiasAdd_grad/tuple/control_dependencyIdentity\default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/Relu_grad/ReluGradh^default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/BiasAdd_grad/tuple/group_deps*
T0*o
_classe
caloc:@default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/Relu_grad/ReluGrad*(
_output_shapes
:џџџџџџџџџ
ш
qdefault_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/BiasAdd_grad/tuple/control_dependency_1Identitybdefault_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/BiasAdd_grad/BiasAddGradh^default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/BiasAdd_grad/tuple/group_deps*
T0*u
_classk
igloc:@default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
О
gdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/BiasAdd_grad/BiasAddGradBiasAddGradadefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/Relu_grad/ReluGrad*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:*
data_formatNHWC
ъ
ldefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/BiasAdd_grad/tuple/group_depsNoOph^default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/BiasAdd_grad/BiasAddGradb^default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/Relu_grad/ReluGrad*&
 _has_manual_control_dependencies(
ћ
tdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/BiasAdd_grad/tuple/control_dependencyIdentityadefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/Relu_grad/ReluGradm^default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/BiasAdd_grad/tuple/group_deps*
T0*t
_classj
hfloc:@default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/Relu_grad/ReluGrad*(
_output_shapes
:џџџџџџџџџ
ќ
vdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/BiasAdd_grad/tuple/control_dependency_1Identitygdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/BiasAdd_grad/BiasAddGradm^default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/BiasAdd_grad/tuple/group_deps*
T0*z
_classp
nlloc:@default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ю
Ydefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/ShapeShape5default_policy/default_policy_Normal_2_3/log_prob/mul*
T0*
_output_shapes
:*
out_type0
а
[default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/Shape_1Shape5default_policy/default_policy_Normal_2_3/log_prob/add*
T0*
_output_shapes
:*
out_type0
ї
idefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/BroadcastGradientArgsBroadcastGradientArgsYdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/Shape[default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
щ
Wdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/SumSum^default_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/tuple/control_dependencyidefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

[default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/ReshapeReshapeWdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/SumYdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
№
Wdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/NegNeg^default_policy/gradients/default_policy/clip_by_value_16/Minimum_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ц
Ydefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/Sum_1SumWdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/Negkdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

]default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/Reshape_1ReshapeYdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/Sum_1[default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
в
ddefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/tuple/group_depsNoOp\^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/Reshape^^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/Reshape_1*&
 _has_manual_control_dependencies(
о
ldefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/tuple/control_dependencyIdentity[default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/Reshapee^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/tuple/group_deps*
T0*n
_classd
b`loc:@default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
ф
ndefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/tuple/control_dependency_1Identity]default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/Reshape_1e^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/tuple/group_deps*
T0*p
_classf
dbloc:@default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

9default_policy/gradients/default_policy/sub_18_grad/ShapeShapedefault_policy/sub_18/x*
T0*
_output_shapes
: *
out_type0

;default_policy/gradients/default_policy/sub_18_grad/Shape_1Shapedefault_policy/pow_4*
T0*
_output_shapes
:*
out_type0

Idefault_policy/gradients/default_policy/sub_18_grad/BroadcastGradientArgsBroadcastGradientArgs9default_policy/gradients/default_policy/sub_18_grad/Shape;default_policy/gradients/default_policy/sub_18_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

7default_policy/gradients/default_policy/sub_18_grad/SumSumLdefault_policy/gradients/default_policy/add_12_grad/tuple/control_dependencyIdefault_policy/gradients/default_policy/sub_18_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

;default_policy/gradients/default_policy/sub_18_grad/ReshapeReshape7default_policy/gradients/default_policy/sub_18_grad/Sum9default_policy/gradients/default_policy/sub_18_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*
_output_shapes
: 
О
7default_policy/gradients/default_policy/sub_18_grad/NegNegLdefault_policy/gradients/default_policy/add_12_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ

9default_policy/gradients/default_policy/sub_18_grad/Sum_1Sum7default_policy/gradients/default_policy/sub_18_grad/NegKdefault_policy/gradients/default_policy/sub_18_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Ј
=default_policy/gradients/default_policy/sub_18_grad/Reshape_1Reshape9default_policy/gradients/default_policy/sub_18_grad/Sum_1;default_policy/gradients/default_policy/sub_18_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
ђ
Ddefault_policy/gradients/default_policy/sub_18_grad/tuple/group_depsNoOp<^default_policy/gradients/default_policy/sub_18_grad/Reshape>^default_policy/gradients/default_policy/sub_18_grad/Reshape_1*&
 _has_manual_control_dependencies(
Э
Ldefault_policy/gradients/default_policy/sub_18_grad/tuple/control_dependencyIdentity;default_policy/gradients/default_policy/sub_18_grad/ReshapeE^default_policy/gradients/default_policy/sub_18_grad/tuple/group_deps*
T0*N
_classD
B@loc:@default_policy/gradients/default_policy/sub_18_grad/Reshape*
_output_shapes
: 
ф
Ndefault_policy/gradients/default_policy/sub_18_grad/tuple/control_dependency_1Identity=default_policy/gradients/default_policy/sub_18_grad/Reshape_1E^default_policy/gradients/default_policy/sub_18_grad/tuple/group_deps*
T0*P
_classF
DBloc:@default_policy/gradients/default_policy/sub_18_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
 
\default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/MatMul_grad/MatMulMatMulodefault_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/BiasAdd_grad/tuple/control_dependencyFdefault_policy/model_1_2/sequential_1/q_hidden_1/MatMul/ReadVariableOp*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(

^default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/MatMul_grad/MatMul_1MatMul5default_policy/model_1_2/sequential_1/q_hidden_0/Reluodefault_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
ж
fdefault_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/MatMul_grad/tuple/group_depsNoOp]^default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/MatMul_grad/MatMul_^default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/MatMul_grad/MatMul_1*&
 _has_manual_control_dependencies(
х
ndefault_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/MatMul_grad/tuple/control_dependencyIdentity\default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/MatMul_grad/MatMulg^default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/MatMul_grad/tuple/group_deps*
T0*o
_classe
caloc:@default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
у
pdefault_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/MatMul_grad/tuple/control_dependency_1Identity^default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/MatMul_grad/MatMul_1g^default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/MatMul_grad/tuple/group_deps*
T0*q
_classg
ecloc:@default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/MatMul_grad/MatMul_1* 
_output_shapes
:

Џ
adefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/MatMul_grad/MatMulMatMultdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/BiasAdd_grad/tuple/control_dependencyKdefault_policy/model_2_2/sequential_2/twin_q_hidden_1/MatMul/ReadVariableOp*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(

cdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/MatMul_grad/MatMul_1MatMul:default_policy/model_2_2/sequential_2/twin_q_hidden_0/Relutdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
х
kdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/MatMul_grad/tuple/group_depsNoOpb^default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/MatMul_grad/MatMuld^default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/MatMul_grad/MatMul_1*&
 _has_manual_control_dependencies(
љ
sdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/MatMul_grad/tuple/control_dependencyIdentityadefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/MatMul_grad/MatMull^default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/MatMul_grad/tuple/group_deps*
T0*t
_classj
hfloc:@default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
ї
udefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/MatMul_grad/tuple/control_dependency_1Identitycdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/MatMul_grad/MatMul_1l^default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/MatMul_grad/tuple/group_deps*
T0*v
_classl
jhloc:@default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/MatMul_grad/MatMul_1* 
_output_shapes
:

Ю
Ydefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/ShapeShape7default_policy/default_policy_Normal_2_3/log_prob/mul/x*
T0*
_output_shapes
: *
out_type0
о
[default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/Shape_1ShapeCdefault_policy/default_policy_Normal_2_3/log_prob/SquaredDifference*
T0*
_output_shapes
:*
out_type0
ї
idefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/BroadcastGradientArgsBroadcastGradientArgsYdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/Shape[default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
У
Wdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/MulMulldefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/tuple/control_dependencyCdefault_policy/default_policy_Normal_2_3/log_prob/SquaredDifference*
T0*'
_output_shapes
:џџџџџџџџџ
т
Wdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/SumSumWdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/Mulidefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
ё
[default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/ReshapeReshapeWdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/SumYdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*
_output_shapes
: 
Й
Ydefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/Mul_1Mul7default_policy/default_policy_Normal_2_3/log_prob/mul/xldefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ш
Ydefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/Sum_1SumYdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/Mul_1kdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

]default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/Reshape_1ReshapeYdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/Sum_1[default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
в
ddefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/tuple/group_depsNoOp\^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/Reshape^^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/Reshape_1*&
 _has_manual_control_dependencies(
Э
ldefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/tuple/control_dependencyIdentity[default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/Reshapee^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/tuple/group_deps*
T0*n
_classd
b`loc:@default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/Reshape*
_output_shapes
: 

ndefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/tuple/control_dependency_1Identity]default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/Reshape_1e^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/tuple/group_deps*
T0*p
_classf
dbloc:@default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/Reshape_1*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
Ю
Ydefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/ShapeShape7default_policy/default_policy_Normal_2_3/log_prob/Const*
T0*
_output_shapes
: *
out_type0
а
[default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/Shape_1Shape5default_policy/default_policy_Normal_2_3/log_prob/Log*
T0*
_output_shapes
:*
out_type0
ї
idefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/BroadcastGradientArgsBroadcastGradientArgsYdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/Shape[default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
љ
Wdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/SumSumndefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/tuple/control_dependency_1idefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
ё
[default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/ReshapeReshapeWdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/SumYdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*
_output_shapes
: 
§
Ydefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/Sum_1Sumndefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/sub_grad/tuple/control_dependency_1kdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

]default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/Reshape_1ReshapeYdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/Sum_1[default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
в
ddefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/tuple/group_depsNoOp\^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/Reshape^^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/Reshape_1*&
 _has_manual_control_dependencies(
Э
ldefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/tuple/control_dependencyIdentity[default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/Reshapee^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/tuple/group_deps*
T0*n
_classd
b`loc:@default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/Reshape*
_output_shapes
: 

ndefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/tuple/control_dependency_1Identity]default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/Reshape_1e^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/tuple/group_deps*
T0*p
_classf
dbloc:@default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/Reshape_1*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ

8default_policy/gradients/default_policy/pow_4_grad/ShapeShapedefault_policy/Tanh_8*
T0*
_output_shapes
:*
out_type0

:default_policy/gradients/default_policy/pow_4_grad/Shape_1Shapedefault_policy/pow_4/y*
T0*
_output_shapes
: *
out_type0

Hdefault_policy/gradients/default_policy/pow_4_grad/BroadcastGradientArgsBroadcastGradientArgs8default_policy/gradients/default_policy/pow_4_grad/Shape:default_policy/gradients/default_policy/pow_4_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
з
6default_policy/gradients/default_policy/pow_4_grad/mulMulNdefault_policy/gradients/default_policy/sub_18_grad/tuple/control_dependency_1default_policy/pow_4/y*
T0*'
_output_shapes
:џџџџџџџџџ
}
8default_policy/gradients/default_policy/pow_4_grad/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
А
6default_policy/gradients/default_policy/pow_4_grad/subSubdefault_policy/pow_4/y8default_policy/gradients/default_policy/pow_4_grad/sub/y*
T0*
_output_shapes
: 
О
6default_policy/gradients/default_policy/pow_4_grad/PowPowdefault_policy/Tanh_86default_policy/gradients/default_policy/pow_4_grad/sub*
T0*'
_output_shapes
:џџџџџџџџџ
с
8default_policy/gradients/default_policy/pow_4_grad/mul_1Mul6default_policy/gradients/default_policy/pow_4_grad/mul6default_policy/gradients/default_policy/pow_4_grad/Pow*
T0*'
_output_shapes
:џџџџџџџџџ

6default_policy/gradients/default_policy/pow_4_grad/SumSum8default_policy/gradients/default_policy/pow_4_grad/mul_1Hdefault_policy/gradients/default_policy/pow_4_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

:default_policy/gradients/default_policy/pow_4_grad/ReshapeReshape6default_policy/gradients/default_policy/pow_4_grad/Sum8default_policy/gradients/default_policy/pow_4_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ

<default_policy/gradients/default_policy/pow_4_grad/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
Ь
:default_policy/gradients/default_policy/pow_4_grad/GreaterGreaterdefault_policy/Tanh_8<default_policy/gradients/default_policy/pow_4_grad/Greater/y*
T0*'
_output_shapes
:џџџџџџџџџ

Bdefault_policy/gradients/default_policy/pow_4_grad/ones_like/ShapeShapedefault_policy/Tanh_8*
T0*
_output_shapes
:*
out_type0

Bdefault_policy/gradients/default_policy/pow_4_grad/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?

<default_policy/gradients/default_policy/pow_4_grad/ones_likeFillBdefault_policy/gradients/default_policy/pow_4_grad/ones_like/ShapeBdefault_policy/gradients/default_policy/pow_4_grad/ones_like/Const*
T0*'
_output_shapes
:џџџџџџџџџ*

index_type0

9default_policy/gradients/default_policy/pow_4_grad/SelectSelect:default_policy/gradients/default_policy/pow_4_grad/Greaterdefault_policy/Tanh_8<default_policy/gradients/default_policy/pow_4_grad/ones_like*
T0*'
_output_shapes
:џџџџџџџџџ
Њ
6default_policy/gradients/default_policy/pow_4_grad/LogLog9default_policy/gradients/default_policy/pow_4_grad/Select*
T0*'
_output_shapes
:џџџџџџџџџ

=default_policy/gradients/default_policy/pow_4_grad/zeros_like	ZerosLikedefault_policy/Tanh_8*
T0*'
_output_shapes
:џџџџџџџџџ
Њ
;default_policy/gradients/default_policy/pow_4_grad/Select_1Select:default_policy/gradients/default_policy/pow_4_grad/Greater6default_policy/gradients/default_policy/pow_4_grad/Log=default_policy/gradients/default_policy/pow_4_grad/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
з
8default_policy/gradients/default_policy/pow_4_grad/mul_2MulNdefault_policy/gradients/default_policy/sub_18_grad/tuple/control_dependency_1default_policy/pow_4*
T0*'
_output_shapes
:џџџџџџџџџ
ш
8default_policy/gradients/default_policy/pow_4_grad/mul_3Mul8default_policy/gradients/default_policy/pow_4_grad/mul_2;default_policy/gradients/default_policy/pow_4_grad/Select_1*
T0*'
_output_shapes
:џџџџџџџџџ

8default_policy/gradients/default_policy/pow_4_grad/Sum_1Sum8default_policy/gradients/default_policy/pow_4_grad/mul_3Jdefault_policy/gradients/default_policy/pow_4_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

<default_policy/gradients/default_policy/pow_4_grad/Reshape_1Reshape8default_policy/gradients/default_policy/pow_4_grad/Sum_1:default_policy/gradients/default_policy/pow_4_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*
_output_shapes
: 
я
Cdefault_policy/gradients/default_policy/pow_4_grad/tuple/group_depsNoOp;^default_policy/gradients/default_policy/pow_4_grad/Reshape=^default_policy/gradients/default_policy/pow_4_grad/Reshape_1*&
 _has_manual_control_dependencies(

Kdefault_policy/gradients/default_policy/pow_4_grad/tuple/control_dependencyIdentity:default_policy/gradients/default_policy/pow_4_grad/ReshapeD^default_policy/gradients/default_policy/pow_4_grad/tuple/group_deps*
T0*M
_classC
A?loc:@default_policy/gradients/default_policy/pow_4_grad/Reshape*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
Я
Mdefault_policy/gradients/default_policy/pow_4_grad/tuple/control_dependency_1Identity<default_policy/gradients/default_policy/pow_4_grad/Reshape_1D^default_policy/gradients/default_policy/pow_4_grad/tuple/group_deps*
T0*O
_classE
CAloc:@default_policy/gradients/default_policy/pow_4_grad/Reshape_1*
_output_shapes
: 
ъ
\default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/Relu_grad/ReluGradReluGradndefault_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_1/MatMul_grad/tuple/control_dependency5default_policy/model_1_2/sequential_1/q_hidden_0/Relu*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:џџџџџџџџџ
љ
adefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/Relu_grad/ReluGradReluGradsdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_1/MatMul_grad/tuple/control_dependency:default_policy/model_2_2/sequential_2/twin_q_hidden_0/Relu*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:џџџџџџџџџ

hdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/scalarConsto^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/tuple/control_dependency_1*
_output_shapes
: *
dtype0*
valueB
 *   @
ј
edefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/MulMulhdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/scalarndefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

edefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/subSub9default_policy/default_policy_Normal_2_3/log_prob/truediv;default_policy/default_policy_Normal_2_3/log_prob/truediv_1o^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ю
gdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/mul_1Muledefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/Muledefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/sub*
T0*'
_output_shapes
:џџџџџџџџџ
р
gdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/ShapeShape9default_policy/default_policy_Normal_2_3/log_prob/truediv*
T0*
_output_shapes
:*
out_type0
ф
idefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/Shape_1Shape;default_policy/default_policy_Normal_2_3/log_prob/truediv_1*
T0*
_output_shapes
:*
out_type0
Ё
wdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsgdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/Shapeidefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

edefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/SumSumgdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/mul_1wdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Ќ
idefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/ReshapeReshapeedefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/Sumgdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ

gdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/Sum_1Sumgdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/mul_1ydefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

kdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/Reshape_1Reshapegdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/Sum_1idefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Г
edefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/NegNegkdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/Reshape_1*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
і
rdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/tuple/group_depsNoOpf^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/Negj^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/Reshape*&
 _has_manual_control_dependencies(

zdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/tuple/control_dependencyIdentityidefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/Reshapes^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/tuple/group_deps*
T0*|
_classr
pnloc:@default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

|default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/tuple/control_dependency_1Identityedefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/Negs^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/tuple/group_deps*
T0*x
_classn
ljloc:@default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/Neg*'
_output_shapes
:џџџџџџџџџ
Ѕ
^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/Log_grad/Reciprocal
Reciprocaldefault_policy/Exp_5o^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
р
Wdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/Log_grad/mulMulndefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/add_grad/tuple/control_dependency_1^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/Log_grad/Reciprocal*
T0*'
_output_shapes
:џџџџџџџџџ
о
<default_policy/gradients/default_policy/Tanh_8_grad/TanhGradTanhGraddefault_policy/Tanh_8Kdefault_policy/gradients/default_policy/pow_4_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
Д
bdefault_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/BiasAdd_grad/BiasAddGradBiasAddGrad\default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/Relu_grad/ReluGrad*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:*
data_formatNHWC
л
gdefault_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/BiasAdd_grad/tuple/group_depsNoOpc^default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/BiasAdd_grad/BiasAddGrad]^default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/Relu_grad/ReluGrad*&
 _has_manual_control_dependencies(
ч
odefault_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/BiasAdd_grad/tuple/control_dependencyIdentity\default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/Relu_grad/ReluGradh^default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/BiasAdd_grad/tuple/group_deps*
T0*o
_classe
caloc:@default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/Relu_grad/ReluGrad*(
_output_shapes
:џџџџџџџџџ
ш
qdefault_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/BiasAdd_grad/tuple/control_dependency_1Identitybdefault_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/BiasAdd_grad/BiasAddGradh^default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/BiasAdd_grad/tuple/group_deps*
T0*u
_classk
igloc:@default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
О
gdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/BiasAdd_grad/BiasAddGradBiasAddGradadefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/Relu_grad/ReluGrad*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:*
data_formatNHWC
ъ
ldefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/BiasAdd_grad/tuple/group_depsNoOph^default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/BiasAdd_grad/BiasAddGradb^default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/Relu_grad/ReluGrad*&
 _has_manual_control_dependencies(
ћ
tdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/BiasAdd_grad/tuple/control_dependencyIdentityadefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/Relu_grad/ReluGradm^default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/BiasAdd_grad/tuple/group_deps*
T0*t
_classj
hfloc:@default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/Relu_grad/ReluGrad*(
_output_shapes
:џџџџџџџџџ
ќ
vdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/BiasAdd_grad/tuple/control_dependency_1Identitygdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/BiasAdd_grad/BiasAddGradm^default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/BiasAdd_grad/tuple/group_deps*
T0*z
_classp
nlloc:@default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Г
]default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/ShapeShapedefault_policy/Atanh_4*
T0*
_output_shapes
:*
out_type0
Г
_default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/Shape_1Shapedefault_policy/Exp_5*
T0*
_output_shapes
:*
out_type0

mdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs]default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/Shape_default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ў
_default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/RealDivRealDivzdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/tuple/control_dependencydefault_policy/Exp_5*
T0*'
_output_shapes
:џџџџџџџџџ
ђ
[default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/SumSum_default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/RealDivmdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

_default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/ReshapeReshape[default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/Sum]default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
Ќ
[default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/NegNegdefault_policy/Atanh_4*
T0*'
_output_shapes
:џџџџџџџџџ

adefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/RealDiv_1RealDiv[default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/Negdefault_policy/Exp_5*
T0*'
_output_shapes
:џџџџџџџџџ

adefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/RealDiv_2RealDivadefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/RealDiv_1default_policy/Exp_5*
T0*'
_output_shapes
:џџџџџџџџџ
ѓ
[default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/mulMulzdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/tuple/control_dependencyadefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:џџџџџџџџџ
ђ
]default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/Sum_1Sum[default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/mulodefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

adefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/Reshape_1Reshape]default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/Sum_1_default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
о
hdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/tuple/group_depsNoOp`^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/Reshapeb^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/Reshape_1*&
 _has_manual_control_dependencies(
ю
pdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/tuple/control_dependencyIdentity_default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/Reshapei^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/tuple/group_deps*
T0*r
_classh
fdloc:@default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
є
rdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/tuple/control_dependency_1Identityadefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/Reshape_1i^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/tuple/group_deps*
T0*t
_classj
hfloc:@default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Е
_default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/ShapeShapedefault_policy/split_2*
T0*
_output_shapes
:*
out_type0
Е
adefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/Shape_1Shapedefault_policy/Exp_5*
T0*
_output_shapes
:*
out_type0

odefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgs_default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/Shapeadefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
В
adefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/RealDivRealDiv|default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/tuple/control_dependency_1default_policy/Exp_5*
T0*'
_output_shapes
:џџџџџџџџџ
ј
]default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/SumSumadefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/RealDivodefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

adefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/ReshapeReshape]default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/Sum_default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
Ў
]default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/NegNegdefault_policy/split_2*
T0*'
_output_shapes
:џџџџџџџџџ

cdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/RealDiv_1RealDiv]default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/Negdefault_policy/Exp_5*
T0*'
_output_shapes
:џџџџџџџџџ

cdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/RealDiv_2RealDivcdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/RealDiv_1default_policy/Exp_5*
T0*'
_output_shapes
:џџџџџџџџџ
љ
]default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/mulMul|default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/SquaredDifference_grad/tuple/control_dependency_1cdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/RealDiv_2*
T0*'
_output_shapes
:џџџџџџџџџ
ј
_default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/Sum_1Sum]default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/mulqdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

cdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/Reshape_1Reshape_default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/Sum_1adefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
ф
jdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/tuple/group_depsNoOpb^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/Reshaped^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/Reshape_1*&
 _has_manual_control_dependencies(
і
rdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/tuple/control_dependencyIdentityadefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/Reshapek^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/tuple/group_deps*
T0*t
_classj
hfloc:@default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
ќ
tdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/tuple/control_dependency_1Identitycdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/Reshape_1k^default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/tuple/group_deps*
T0*v
_classl
jhloc:@default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
 
\default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/MatMul_grad/MatMulMatMulodefault_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/BiasAdd_grad/tuple/control_dependencyFdefault_policy/model_1_2/sequential_1/q_hidden_0/MatMul/ReadVariableOp*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(

^default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/MatMul_grad/MatMul_1MatMul8default_policy/model_1_2/sequential_1/concatenate/concatodefault_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
ж
fdefault_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/MatMul_grad/tuple/group_depsNoOp]^default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/MatMul_grad/MatMul_^default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/MatMul_grad/MatMul_1*&
 _has_manual_control_dependencies(
х
ndefault_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/MatMul_grad/tuple/control_dependencyIdentity\default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/MatMul_grad/MatMulg^default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/MatMul_grad/tuple/group_deps*
T0*o
_classe
caloc:@default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
у
pdefault_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/MatMul_grad/tuple/control_dependency_1Identity^default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/MatMul_grad/MatMul_1g^default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/MatMul_grad/tuple/group_deps*
T0*q
_classg
ecloc:@default_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/MatMul_grad/MatMul_1* 
_output_shapes
:

Џ
adefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/MatMul_grad/MatMulMatMultdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/BiasAdd_grad/tuple/control_dependencyKdefault_policy/model_2_2/sequential_2/twin_q_hidden_0/MatMul/ReadVariableOp*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(

cdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/MatMul_grad/MatMul_1MatMul:default_policy/model_2_2/sequential_2/concatenate_1/concattdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
х
kdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/MatMul_grad/tuple/group_depsNoOpb^default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/MatMul_grad/MatMuld^default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/MatMul_grad/MatMul_1*&
 _has_manual_control_dependencies(
љ
sdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/MatMul_grad/tuple/control_dependencyIdentityadefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/MatMul_grad/MatMull^default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/MatMul_grad/tuple/group_deps*
T0*t
_classj
hfloc:@default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
ї
udefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/MatMul_grad/tuple/control_dependency_1Identitycdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/MatMul_grad/MatMul_1l^default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/MatMul_grad/tuple/group_deps*
T0*v
_classl
jhloc:@default_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/MatMul_grad/MatMul_1* 
_output_shapes
:


default_policy/gradients/AddNAddN<default_policy/gradients/default_policy/Tanh_8_grad/TanhGradpdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/tuple/control_dependency*
N*
T0*O
_classE
CAloc:@default_policy/gradients/default_policy/Tanh_8_grad/TanhGrad*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
И
;default_policy/gradients/default_policy/Atanh_4_grad/SquareSquaredefault_policy/clip_by_value_15^default_policy/gradients/AddN*
T0*'
_output_shapes
:џџџџџџџџџ

:default_policy/gradients/default_policy/Atanh_4_grad/ConstConst^default_policy/gradients/AddN*
_output_shapes
: *
dtype0*
valueB
 *  ?
ъ
8default_policy/gradients/default_policy/Atanh_4_grad/SubSub:default_policy/gradients/default_policy/Atanh_4_grad/Const;default_policy/gradients/default_policy/Atanh_4_grad/Square*
T0*'
_output_shapes
:џџџџџџџџџ
Й
?default_policy/gradients/default_policy/Atanh_4_grad/Reciprocal
Reciprocal8default_policy/gradients/default_policy/Atanh_4_grad/Sub*
T0*'
_output_shapes
:џџџџџџџџџ
б
8default_policy/gradients/default_policy/Atanh_4_grad/mulMuldefault_policy/gradients/AddN?default_policy/gradients/default_policy/Atanh_4_grad/Reciprocal*
T0*'
_output_shapes
:џџџџџџџџџ

[default_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
Ѓ
Zdefault_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/modFloorMod=default_policy/model_1_2/sequential_1/concatenate/concat/axis[default_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/Rank*
T0*
_output_shapes
: 
Ж
\default_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/ShapeShapedefault_policy/observation*
T0*
_output_shapes
:*
out_type0
ш
]default_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/ShapeNShapeNdefault_policy/observationdefault_policy/clip_by_value_12*
N*
T0* 
_output_shapes
::*
out_type0
К
cdefault_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/ConcatOffsetConcatOffsetZdefault_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/mod]default_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/ShapeN_default_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/ShapeN:1*
N* 
_output_shapes
::

\default_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/SliceSlicendefault_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/MatMul_grad/tuple/control_dependencycdefault_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/ConcatOffset]default_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/ShapeN*
Index0*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:џџџџџџџџџ

^default_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/Slice_1Slicendefault_policy/gradients/default_policy/model_1_2/sequential_1/q_hidden_0/MatMul_grad/tuple/control_dependencyedefault_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/ConcatOffset:1_default_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/ShapeN:1*
Index0*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
з
gdefault_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/tuple/group_depsNoOp]^default_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/Slice_^default_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/Slice_1*&
 _has_manual_control_dependencies(
ч
odefault_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/tuple/control_dependencyIdentity\default_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/Sliceh^default_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/tuple/group_deps*
T0*o
_classe
caloc:@default_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/Slice*(
_output_shapes
:џџџџџџџџџ
ь
qdefault_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/tuple/control_dependency_1Identity^default_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/Slice_1h^default_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/tuple/group_deps*
T0*q
_classg
ecloc:@default_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/Slice_1*'
_output_shapes
:џџџџџџџџџ

]default_policy/gradients/default_policy/model_2_2/sequential_2/concatenate_1/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
Љ
\default_policy/gradients/default_policy/model_2_2/sequential_2/concatenate_1/concat_grad/modFloorMod?default_policy/model_2_2/sequential_2/concatenate_1/concat/axis]default_policy/gradients/default_policy/model_2_2/sequential_2/concatenate_1/concat_grad/Rank*
T0*
_output_shapes
: 
И
^default_policy/gradients/default_policy/model_2_2/sequential_2/concatenate_1/concat_grad/ShapeShapedefault_policy/observation*
T0*
_output_shapes
:*
out_type0
ъ
_default_policy/gradients/default_policy/model_2_2/sequential_2/concatenate_1/concat_grad/ShapeNShapeNdefault_policy/observationdefault_policy/clip_by_value_12*
N*
T0* 
_output_shapes
::*
out_type0
Т
edefault_policy/gradients/default_policy/model_2_2/sequential_2/concatenate_1/concat_grad/ConcatOffsetConcatOffset\default_policy/gradients/default_policy/model_2_2/sequential_2/concatenate_1/concat_grad/mod_default_policy/gradients/default_policy/model_2_2/sequential_2/concatenate_1/concat_grad/ShapeNadefault_policy/gradients/default_policy/model_2_2/sequential_2/concatenate_1/concat_grad/ShapeN:1*
N* 
_output_shapes
::

^default_policy/gradients/default_policy/model_2_2/sequential_2/concatenate_1/concat_grad/SliceSlicesdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/MatMul_grad/tuple/control_dependencyedefault_policy/gradients/default_policy/model_2_2/sequential_2/concatenate_1/concat_grad/ConcatOffset_default_policy/gradients/default_policy/model_2_2/sequential_2/concatenate_1/concat_grad/ShapeN*
Index0*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:џџџџџџџџџ

`default_policy/gradients/default_policy/model_2_2/sequential_2/concatenate_1/concat_grad/Slice_1Slicesdefault_policy/gradients/default_policy/model_2_2/sequential_2/twin_q_hidden_0/MatMul_grad/tuple/control_dependencygdefault_policy/gradients/default_policy/model_2_2/sequential_2/concatenate_1/concat_grad/ConcatOffset:1adefault_policy/gradients/default_policy/model_2_2/sequential_2/concatenate_1/concat_grad/ShapeN:1*
Index0*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
н
idefault_policy/gradients/default_policy/model_2_2/sequential_2/concatenate_1/concat_grad/tuple/group_depsNoOp_^default_policy/gradients/default_policy/model_2_2/sequential_2/concatenate_1/concat_grad/Slicea^default_policy/gradients/default_policy/model_2_2/sequential_2/concatenate_1/concat_grad/Slice_1*&
 _has_manual_control_dependencies(
я
qdefault_policy/gradients/default_policy/model_2_2/sequential_2/concatenate_1/concat_grad/tuple/control_dependencyIdentity^default_policy/gradients/default_policy/model_2_2/sequential_2/concatenate_1/concat_grad/Slicej^default_policy/gradients/default_policy/model_2_2/sequential_2/concatenate_1/concat_grad/tuple/group_deps*
T0*q
_classg
ecloc:@default_policy/gradients/default_policy/model_2_2/sequential_2/concatenate_1/concat_grad/Slice*(
_output_shapes
:џџџџџџџџџ
є
sdefault_policy/gradients/default_policy/model_2_2/sequential_2/concatenate_1/concat_grad/tuple/control_dependency_1Identity`default_policy/gradients/default_policy/model_2_2/sequential_2/concatenate_1/concat_grad/Slice_1j^default_policy/gradients/default_policy/model_2_2/sequential_2/concatenate_1/concat_grad/tuple/group_deps*
T0*s
_classi
geloc:@default_policy/gradients/default_policy/model_2_2/sequential_2/concatenate_1/concat_grad/Slice_1*'
_output_shapes
:џџџџџџџџџ
Њ
Cdefault_policy/gradients/default_policy/clip_by_value_15_grad/ShapeShape'default_policy/clip_by_value_15/Minimum*
T0*
_output_shapes
:*
out_type0

Edefault_policy/gradients/default_policy/clip_by_value_15_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
С
Hdefault_policy/gradients/default_policy/clip_by_value_15_grad/zeros_like	ZerosLike8default_policy/gradients/default_policy/Atanh_4_grad/mul*
T0*'
_output_shapes
:џџџџџџџџџ
и
Jdefault_policy/gradients/default_policy/clip_by_value_15_grad/GreaterEqualGreaterEqual'default_policy/clip_by_value_15/Minimum!default_policy/clip_by_value_15/y*
T0*'
_output_shapes
:џџџџџџџџџ
Е
Sdefault_policy/gradients/default_policy/clip_by_value_15_grad/BroadcastGradientArgsBroadcastGradientArgsCdefault_policy/gradients/default_policy/clip_by_value_15_grad/ShapeEdefault_policy/gradients/default_policy/clip_by_value_15_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
д
Fdefault_policy/gradients/default_policy/clip_by_value_15_grad/SelectV2SelectV2Jdefault_policy/gradients/default_policy/clip_by_value_15_grad/GreaterEqual8default_policy/gradients/default_policy/Atanh_4_grad/mulHdefault_policy/gradients/default_policy/clip_by_value_15_grad/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
Ѕ
Adefault_policy/gradients/default_policy/clip_by_value_15_grad/SumSumFdefault_policy/gradients/default_policy/clip_by_value_15_grad/SelectV2Sdefault_policy/gradients/default_policy/clip_by_value_15_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Р
Edefault_policy/gradients/default_policy/clip_by_value_15_grad/ReshapeReshapeAdefault_policy/gradients/default_policy/clip_by_value_15_grad/SumCdefault_policy/gradients/default_policy/clip_by_value_15_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
ж
Hdefault_policy/gradients/default_policy/clip_by_value_15_grad/SelectV2_1SelectV2Jdefault_policy/gradients/default_policy/clip_by_value_15_grad/GreaterEqualHdefault_policy/gradients/default_policy/clip_by_value_15_grad/zeros_like8default_policy/gradients/default_policy/Atanh_4_grad/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Ћ
Cdefault_policy/gradients/default_policy/clip_by_value_15_grad/Sum_1SumHdefault_policy/gradients/default_policy/clip_by_value_15_grad/SelectV2_1Udefault_policy/gradients/default_policy/clip_by_value_15_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Е
Gdefault_policy/gradients/default_policy/clip_by_value_15_grad/Reshape_1ReshapeCdefault_policy/gradients/default_policy/clip_by_value_15_grad/Sum_1Edefault_policy/gradients/default_policy/clip_by_value_15_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*
_output_shapes
: 

Ndefault_policy/gradients/default_policy/clip_by_value_15_grad/tuple/group_depsNoOpF^default_policy/gradients/default_policy/clip_by_value_15_grad/ReshapeH^default_policy/gradients/default_policy/clip_by_value_15_grad/Reshape_1*&
 _has_manual_control_dependencies(

Vdefault_policy/gradients/default_policy/clip_by_value_15_grad/tuple/control_dependencyIdentityEdefault_policy/gradients/default_policy/clip_by_value_15_grad/ReshapeO^default_policy/gradients/default_policy/clip_by_value_15_grad/tuple/group_deps*
T0*X
_classN
LJloc:@default_policy/gradients/default_policy/clip_by_value_15_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
ћ
Xdefault_policy/gradients/default_policy/clip_by_value_15_grad/tuple/control_dependency_1IdentityGdefault_policy/gradients/default_policy/clip_by_value_15_grad/Reshape_1O^default_policy/gradients/default_policy/clip_by_value_15_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@default_policy/gradients/default_policy/clip_by_value_15_grad/Reshape_1*
_output_shapes
: 
 
Kdefault_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/ShapeShapedefault_policy/sub_17*
T0*
_output_shapes
:*
out_type0

Mdefault_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
ч
Pdefault_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/zeros_like	ZerosLikeVdefault_policy/gradients/default_policy/clip_by_value_15_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
а
Odefault_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/LessEqual	LessEqualdefault_policy/sub_17)default_policy/clip_by_value_15/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
Э
[default_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsKdefault_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/ShapeMdefault_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Ndefault_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/SelectV2SelectV2Odefault_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/LessEqualVdefault_policy/gradients/default_policy/clip_by_value_15_grad/tuple/control_dependencyPdefault_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
Н
Idefault_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/SumSumNdefault_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/SelectV2[default_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
и
Mdefault_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/ReshapeReshapeIdefault_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/SumKdefault_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ

Pdefault_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/SelectV2_1SelectV2Odefault_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/LessEqualPdefault_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/zeros_likeVdefault_policy/gradients/default_policy/clip_by_value_15_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
У
Kdefault_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/Sum_1SumPdefault_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/SelectV2_1]default_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Э
Odefault_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/Reshape_1ReshapeKdefault_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/Sum_1Mdefault_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*
_output_shapes
: 
Ј
Vdefault_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/tuple/group_depsNoOpN^default_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/ReshapeP^default_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/Reshape_1*&
 _has_manual_control_dependencies(
І
^default_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/tuple/control_dependencyIdentityMdefault_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/ReshapeW^default_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/tuple/group_deps*
T0*`
_classV
TRloc:@default_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

`default_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/tuple/control_dependency_1IdentityOdefault_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/Reshape_1W^default_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/tuple/group_deps*
T0*b
_classX
VTloc:@default_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/Reshape_1*
_output_shapes
: 

9default_policy/gradients/default_policy/sub_17_grad/ShapeShapedefault_policy/mul_8*
T0*
_output_shapes
:*
out_type0

;default_policy/gradients/default_policy/sub_17_grad/Shape_1Shapedefault_policy/sub_17/y*
T0*
_output_shapes
: *
out_type0

Idefault_policy/gradients/default_policy/sub_17_grad/BroadcastGradientArgsBroadcastGradientArgs9default_policy/gradients/default_policy/sub_17_grad/Shape;default_policy/gradients/default_policy/sub_17_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Љ
7default_policy/gradients/default_policy/sub_17_grad/SumSum^default_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/tuple/control_dependencyIdefault_policy/gradients/default_policy/sub_17_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Ђ
;default_policy/gradients/default_policy/sub_17_grad/ReshapeReshape7default_policy/gradients/default_policy/sub_17_grad/Sum9default_policy/gradients/default_policy/sub_17_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
а
7default_policy/gradients/default_policy/sub_17_grad/NegNeg^default_policy/gradients/default_policy/clip_by_value_15/Minimum_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ

9default_policy/gradients/default_policy/sub_17_grad/Sum_1Sum7default_policy/gradients/default_policy/sub_17_grad/NegKdefault_policy/gradients/default_policy/sub_17_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

=default_policy/gradients/default_policy/sub_17_grad/Reshape_1Reshape9default_policy/gradients/default_policy/sub_17_grad/Sum_1;default_policy/gradients/default_policy/sub_17_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*
_output_shapes
: 
ђ
Ddefault_policy/gradients/default_policy/sub_17_grad/tuple/group_depsNoOp<^default_policy/gradients/default_policy/sub_17_grad/Reshape>^default_policy/gradients/default_policy/sub_17_grad/Reshape_1*&
 _has_manual_control_dependencies(
о
Ldefault_policy/gradients/default_policy/sub_17_grad/tuple/control_dependencyIdentity;default_policy/gradients/default_policy/sub_17_grad/ReshapeE^default_policy/gradients/default_policy/sub_17_grad/tuple/group_deps*
T0*N
_classD
B@loc:@default_policy/gradients/default_policy/sub_17_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
г
Ndefault_policy/gradients/default_policy/sub_17_grad/tuple/control_dependency_1Identity=default_policy/gradients/default_policy/sub_17_grad/Reshape_1E^default_policy/gradients/default_policy/sub_17_grad/tuple/group_deps*
T0*P
_classF
DBloc:@default_policy/gradients/default_policy/sub_17_grad/Reshape_1*
_output_shapes
: 

8default_policy/gradients/default_policy/mul_8_grad/ShapeShapedefault_policy/truediv_8*
T0*
_output_shapes
:*
out_type0

:default_policy/gradients/default_policy/mul_8_grad/Shape_1Shapedefault_policy/mul_8/y*
T0*
_output_shapes
: *
out_type0

Hdefault_policy/gradients/default_policy/mul_8_grad/BroadcastGradientArgsBroadcastGradientArgs8default_policy/gradients/default_policy/mul_8_grad/Shape:default_policy/gradients/default_policy/mul_8_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
е
6default_policy/gradients/default_policy/mul_8_grad/MulMulLdefault_policy/gradients/default_policy/sub_17_grad/tuple/control_dependencydefault_policy/mul_8/y*
T0*'
_output_shapes
:џџџџџџџџџ
џ
6default_policy/gradients/default_policy/mul_8_grad/SumSum6default_policy/gradients/default_policy/mul_8_grad/MulHdefault_policy/gradients/default_policy/mul_8_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

:default_policy/gradients/default_policy/mul_8_grad/ReshapeReshape6default_policy/gradients/default_policy/mul_8_grad/Sum8default_policy/gradients/default_policy/mul_8_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
й
8default_policy/gradients/default_policy/mul_8_grad/Mul_1Muldefault_policy/truediv_8Ldefault_policy/gradients/default_policy/sub_17_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ

8default_policy/gradients/default_policy/mul_8_grad/Sum_1Sum8default_policy/gradients/default_policy/mul_8_grad/Mul_1Jdefault_policy/gradients/default_policy/mul_8_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

<default_policy/gradients/default_policy/mul_8_grad/Reshape_1Reshape8default_policy/gradients/default_policy/mul_8_grad/Sum_1:default_policy/gradients/default_policy/mul_8_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*
_output_shapes
: 
я
Cdefault_policy/gradients/default_policy/mul_8_grad/tuple/group_depsNoOp;^default_policy/gradients/default_policy/mul_8_grad/Reshape=^default_policy/gradients/default_policy/mul_8_grad/Reshape_1*&
 _has_manual_control_dependencies(
к
Kdefault_policy/gradients/default_policy/mul_8_grad/tuple/control_dependencyIdentity:default_policy/gradients/default_policy/mul_8_grad/ReshapeD^default_policy/gradients/default_policy/mul_8_grad/tuple/group_deps*
T0*M
_classC
A?loc:@default_policy/gradients/default_policy/mul_8_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Я
Mdefault_policy/gradients/default_policy/mul_8_grad/tuple/control_dependency_1Identity<default_policy/gradients/default_policy/mul_8_grad/Reshape_1D^default_policy/gradients/default_policy/mul_8_grad/tuple/group_deps*
T0*O
_classE
CAloc:@default_policy/gradients/default_policy/mul_8_grad/Reshape_1*
_output_shapes
: 

<default_policy/gradients/default_policy/truediv_8_grad/ShapeShapedefault_policy/sub_16*
T0*
_output_shapes
:*
out_type0

>default_policy/gradients/default_policy/truediv_8_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
 
Ldefault_policy/gradients/default_policy/truediv_8_grad/BroadcastGradientArgsBroadcastGradientArgs<default_policy/gradients/default_policy/truediv_8_grad/Shape>default_policy/gradients/default_policy/truediv_8_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ф
>default_policy/gradients/default_policy/truediv_8_grad/RealDivRealDivKdefault_policy/gradients/default_policy/mul_8_grad/tuple/control_dependencydefault_policy/truediv_8/y*
T0*'
_output_shapes
:џџџџџџџџџ

:default_policy/gradients/default_policy/truediv_8_grad/SumSum>default_policy/gradients/default_policy/truediv_8_grad/RealDivLdefault_policy/gradients/default_policy/truediv_8_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Ћ
>default_policy/gradients/default_policy/truediv_8_grad/ReshapeReshape:default_policy/gradients/default_policy/truediv_8_grad/Sum<default_policy/gradients/default_policy/truediv_8_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ

:default_policy/gradients/default_policy/truediv_8_grad/NegNegdefault_policy/sub_16*
T0*'
_output_shapes
:џџџџџџџџџ
е
@default_policy/gradients/default_policy/truediv_8_grad/RealDiv_1RealDiv:default_policy/gradients/default_policy/truediv_8_grad/Negdefault_policy/truediv_8/y*
T0*'
_output_shapes
:џџџџџџџџџ
л
@default_policy/gradients/default_policy/truediv_8_grad/RealDiv_2RealDiv@default_policy/gradients/default_policy/truediv_8_grad/RealDiv_1default_policy/truediv_8/y*
T0*'
_output_shapes
:џџџџџџџџџ

:default_policy/gradients/default_policy/truediv_8_grad/mulMulKdefault_policy/gradients/default_policy/mul_8_grad/tuple/control_dependency@default_policy/gradients/default_policy/truediv_8_grad/RealDiv_2*
T0*'
_output_shapes
:џџџџџџџџџ

<default_policy/gradients/default_policy/truediv_8_grad/Sum_1Sum:default_policy/gradients/default_policy/truediv_8_grad/mulNdefault_policy/gradients/default_policy/truediv_8_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
 
@default_policy/gradients/default_policy/truediv_8_grad/Reshape_1Reshape<default_policy/gradients/default_policy/truediv_8_grad/Sum_1>default_policy/gradients/default_policy/truediv_8_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*
_output_shapes
: 
ћ
Gdefault_policy/gradients/default_policy/truediv_8_grad/tuple/group_depsNoOp?^default_policy/gradients/default_policy/truediv_8_grad/ReshapeA^default_policy/gradients/default_policy/truediv_8_grad/Reshape_1*&
 _has_manual_control_dependencies(
ъ
Odefault_policy/gradients/default_policy/truediv_8_grad/tuple/control_dependencyIdentity>default_policy/gradients/default_policy/truediv_8_grad/ReshapeH^default_policy/gradients/default_policy/truediv_8_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@default_policy/gradients/default_policy/truediv_8_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
п
Qdefault_policy/gradients/default_policy/truediv_8_grad/tuple/control_dependency_1Identity@default_policy/gradients/default_policy/truediv_8_grad/Reshape_1H^default_policy/gradients/default_policy/truediv_8_grad/tuple/group_deps*
T0*S
_classI
GEloc:@default_policy/gradients/default_policy/truediv_8_grad/Reshape_1*
_output_shapes
: 

9default_policy/gradients/default_policy/sub_16_grad/ShapeShapedefault_policy/clip_by_value_12*
T0*
_output_shapes
:*
out_type0

;default_policy/gradients/default_policy/sub_16_grad/Shape_1Shapedefault_policy/sub_16/y*
T0*
_output_shapes
: *
out_type0

Idefault_policy/gradients/default_policy/sub_16_grad/BroadcastGradientArgsBroadcastGradientArgs9default_policy/gradients/default_policy/sub_16_grad/Shape;default_policy/gradients/default_policy/sub_16_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

7default_policy/gradients/default_policy/sub_16_grad/SumSumOdefault_policy/gradients/default_policy/truediv_8_grad/tuple/control_dependencyIdefault_policy/gradients/default_policy/sub_16_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Ђ
;default_policy/gradients/default_policy/sub_16_grad/ReshapeReshape7default_policy/gradients/default_policy/sub_16_grad/Sum9default_policy/gradients/default_policy/sub_16_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
С
7default_policy/gradients/default_policy/sub_16_grad/NegNegOdefault_policy/gradients/default_policy/truediv_8_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ

9default_policy/gradients/default_policy/sub_16_grad/Sum_1Sum7default_policy/gradients/default_policy/sub_16_grad/NegKdefault_policy/gradients/default_policy/sub_16_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

=default_policy/gradients/default_policy/sub_16_grad/Reshape_1Reshape9default_policy/gradients/default_policy/sub_16_grad/Sum_1;default_policy/gradients/default_policy/sub_16_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*
_output_shapes
: 
ђ
Ddefault_policy/gradients/default_policy/sub_16_grad/tuple/group_depsNoOp<^default_policy/gradients/default_policy/sub_16_grad/Reshape>^default_policy/gradients/default_policy/sub_16_grad/Reshape_1*&
 _has_manual_control_dependencies(
о
Ldefault_policy/gradients/default_policy/sub_16_grad/tuple/control_dependencyIdentity;default_policy/gradients/default_policy/sub_16_grad/ReshapeE^default_policy/gradients/default_policy/sub_16_grad/tuple/group_deps*
T0*N
_classD
B@loc:@default_policy/gradients/default_policy/sub_16_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
г
Ndefault_policy/gradients/default_policy/sub_16_grad/tuple/control_dependency_1Identity=default_policy/gradients/default_policy/sub_16_grad/Reshape_1E^default_policy/gradients/default_policy/sub_16_grad/tuple/group_deps*
T0*P
_classF
DBloc:@default_policy/gradients/default_policy/sub_16_grad/Reshape_1*
_output_shapes
: 

default_policy/gradients/AddN_1AddNqdefault_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/tuple/control_dependency_1sdefault_policy/gradients/default_policy/model_2_2/sequential_2/concatenate_1/concat_grad/tuple/control_dependency_1Ldefault_policy/gradients/default_policy/sub_16_grad/tuple/control_dependency*
N*
T0*q
_classg
ecloc:@default_policy/gradients/default_policy/model_1_2/sequential_1/concatenate/concat_grad/Slice_1*'
_output_shapes
:џџџџџџџџџ
Њ
Cdefault_policy/gradients/default_policy/clip_by_value_12_grad/ShapeShape'default_policy/clip_by_value_12/Minimum*
T0*
_output_shapes
:*
out_type0

Edefault_policy/gradients/default_policy/clip_by_value_12_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
Ј
Hdefault_policy/gradients/default_policy/clip_by_value_12_grad/zeros_like	ZerosLikedefault_policy/gradients/AddN_1*
T0*'
_output_shapes
:џџџџџџџџџ
и
Jdefault_policy/gradients/default_policy/clip_by_value_12_grad/GreaterEqualGreaterEqual'default_policy/clip_by_value_12/Minimum!default_policy/clip_by_value_12/y*
T0*'
_output_shapes
:џџџџџџџџџ
Е
Sdefault_policy/gradients/default_policy/clip_by_value_12_grad/BroadcastGradientArgsBroadcastGradientArgsCdefault_policy/gradients/default_policy/clip_by_value_12_grad/ShapeEdefault_policy/gradients/default_policy/clip_by_value_12_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Л
Fdefault_policy/gradients/default_policy/clip_by_value_12_grad/SelectV2SelectV2Jdefault_policy/gradients/default_policy/clip_by_value_12_grad/GreaterEqualdefault_policy/gradients/AddN_1Hdefault_policy/gradients/default_policy/clip_by_value_12_grad/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
Ѕ
Adefault_policy/gradients/default_policy/clip_by_value_12_grad/SumSumFdefault_policy/gradients/default_policy/clip_by_value_12_grad/SelectV2Sdefault_policy/gradients/default_policy/clip_by_value_12_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Р
Edefault_policy/gradients/default_policy/clip_by_value_12_grad/ReshapeReshapeAdefault_policy/gradients/default_policy/clip_by_value_12_grad/SumCdefault_policy/gradients/default_policy/clip_by_value_12_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
Н
Hdefault_policy/gradients/default_policy/clip_by_value_12_grad/SelectV2_1SelectV2Jdefault_policy/gradients/default_policy/clip_by_value_12_grad/GreaterEqualHdefault_policy/gradients/default_policy/clip_by_value_12_grad/zeros_likedefault_policy/gradients/AddN_1*
T0*'
_output_shapes
:џџџџџџџџџ
Ћ
Cdefault_policy/gradients/default_policy/clip_by_value_12_grad/Sum_1SumHdefault_policy/gradients/default_policy/clip_by_value_12_grad/SelectV2_1Udefault_policy/gradients/default_policy/clip_by_value_12_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Е
Gdefault_policy/gradients/default_policy/clip_by_value_12_grad/Reshape_1ReshapeCdefault_policy/gradients/default_policy/clip_by_value_12_grad/Sum_1Edefault_policy/gradients/default_policy/clip_by_value_12_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*
_output_shapes
: 

Ndefault_policy/gradients/default_policy/clip_by_value_12_grad/tuple/group_depsNoOpF^default_policy/gradients/default_policy/clip_by_value_12_grad/ReshapeH^default_policy/gradients/default_policy/clip_by_value_12_grad/Reshape_1*&
 _has_manual_control_dependencies(

Vdefault_policy/gradients/default_policy/clip_by_value_12_grad/tuple/control_dependencyIdentityEdefault_policy/gradients/default_policy/clip_by_value_12_grad/ReshapeO^default_policy/gradients/default_policy/clip_by_value_12_grad/tuple/group_deps*
T0*X
_classN
LJloc:@default_policy/gradients/default_policy/clip_by_value_12_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
ћ
Xdefault_policy/gradients/default_policy/clip_by_value_12_grad/tuple/control_dependency_1IdentityGdefault_policy/gradients/default_policy/clip_by_value_12_grad/Reshape_1O^default_policy/gradients/default_policy/clip_by_value_12_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@default_policy/gradients/default_policy/clip_by_value_12_grad/Reshape_1*
_output_shapes
: 
 
Kdefault_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/ShapeShapedefault_policy/add_10*
T0*
_output_shapes
:*
out_type0

Mdefault_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
ч
Pdefault_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/zeros_like	ZerosLikeVdefault_policy/gradients/default_policy/clip_by_value_12_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
а
Odefault_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/LessEqual	LessEqualdefault_policy/add_10)default_policy/clip_by_value_12/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
Э
[default_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsKdefault_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/ShapeMdefault_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Ndefault_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/SelectV2SelectV2Odefault_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/LessEqualVdefault_policy/gradients/default_policy/clip_by_value_12_grad/tuple/control_dependencyPdefault_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
Н
Idefault_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/SumSumNdefault_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/SelectV2[default_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
и
Mdefault_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/ReshapeReshapeIdefault_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/SumKdefault_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ

Pdefault_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/SelectV2_1SelectV2Odefault_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/LessEqualPdefault_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/zeros_likeVdefault_policy/gradients/default_policy/clip_by_value_12_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
У
Kdefault_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/Sum_1SumPdefault_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/SelectV2_1]default_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Э
Odefault_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/Reshape_1ReshapeKdefault_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/Sum_1Mdefault_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*
_output_shapes
: 
Ј
Vdefault_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/tuple/group_depsNoOpN^default_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/ReshapeP^default_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/Reshape_1*&
 _has_manual_control_dependencies(
І
^default_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/tuple/control_dependencyIdentityMdefault_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/ReshapeW^default_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/tuple/group_deps*
T0*`
_classV
TRloc:@default_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

`default_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/tuple/control_dependency_1IdentityOdefault_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/Reshape_1W^default_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/tuple/group_deps*
T0*b
_classX
VTloc:@default_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/Reshape_1*
_output_shapes
: 

9default_policy/gradients/default_policy/add_10_grad/ShapeShapedefault_policy/mul_6*
T0*
_output_shapes
:*
out_type0

;default_policy/gradients/default_policy/add_10_grad/Shape_1Shapedefault_policy/add_10/y*
T0*
_output_shapes
: *
out_type0

Idefault_policy/gradients/default_policy/add_10_grad/BroadcastGradientArgsBroadcastGradientArgs9default_policy/gradients/default_policy/add_10_grad/Shape;default_policy/gradients/default_policy/add_10_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Љ
7default_policy/gradients/default_policy/add_10_grad/SumSum^default_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/tuple/control_dependencyIdefault_policy/gradients/default_policy/add_10_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Ђ
;default_policy/gradients/default_policy/add_10_grad/ReshapeReshape7default_policy/gradients/default_policy/add_10_grad/Sum9default_policy/gradients/default_policy/add_10_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
­
9default_policy/gradients/default_policy/add_10_grad/Sum_1Sum^default_policy/gradients/default_policy/clip_by_value_12/Minimum_grad/tuple/control_dependencyKdefault_policy/gradients/default_policy/add_10_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

=default_policy/gradients/default_policy/add_10_grad/Reshape_1Reshape9default_policy/gradients/default_policy/add_10_grad/Sum_1;default_policy/gradients/default_policy/add_10_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*
_output_shapes
: 
ђ
Ddefault_policy/gradients/default_policy/add_10_grad/tuple/group_depsNoOp<^default_policy/gradients/default_policy/add_10_grad/Reshape>^default_policy/gradients/default_policy/add_10_grad/Reshape_1*&
 _has_manual_control_dependencies(
о
Ldefault_policy/gradients/default_policy/add_10_grad/tuple/control_dependencyIdentity;default_policy/gradients/default_policy/add_10_grad/ReshapeE^default_policy/gradients/default_policy/add_10_grad/tuple/group_deps*
T0*N
_classD
B@loc:@default_policy/gradients/default_policy/add_10_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
г
Ndefault_policy/gradients/default_policy/add_10_grad/tuple/control_dependency_1Identity=default_policy/gradients/default_policy/add_10_grad/Reshape_1E^default_policy/gradients/default_policy/add_10_grad/tuple/group_deps*
T0*P
_classF
DBloc:@default_policy/gradients/default_policy/add_10_grad/Reshape_1*
_output_shapes
: 

8default_policy/gradients/default_policy/mul_6_grad/ShapeShapedefault_policy/truediv_6*
T0*
_output_shapes
:*
out_type0

:default_policy/gradients/default_policy/mul_6_grad/Shape_1Shapedefault_policy/mul_6/y*
T0*
_output_shapes
: *
out_type0

Hdefault_policy/gradients/default_policy/mul_6_grad/BroadcastGradientArgsBroadcastGradientArgs8default_policy/gradients/default_policy/mul_6_grad/Shape:default_policy/gradients/default_policy/mul_6_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
е
6default_policy/gradients/default_policy/mul_6_grad/MulMulLdefault_policy/gradients/default_policy/add_10_grad/tuple/control_dependencydefault_policy/mul_6/y*
T0*'
_output_shapes
:џџџџџџџџџ
џ
6default_policy/gradients/default_policy/mul_6_grad/SumSum6default_policy/gradients/default_policy/mul_6_grad/MulHdefault_policy/gradients/default_policy/mul_6_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

:default_policy/gradients/default_policy/mul_6_grad/ReshapeReshape6default_policy/gradients/default_policy/mul_6_grad/Sum8default_policy/gradients/default_policy/mul_6_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
й
8default_policy/gradients/default_policy/mul_6_grad/Mul_1Muldefault_policy/truediv_6Ldefault_policy/gradients/default_policy/add_10_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ

8default_policy/gradients/default_policy/mul_6_grad/Sum_1Sum8default_policy/gradients/default_policy/mul_6_grad/Mul_1Jdefault_policy/gradients/default_policy/mul_6_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

<default_policy/gradients/default_policy/mul_6_grad/Reshape_1Reshape8default_policy/gradients/default_policy/mul_6_grad/Sum_1:default_policy/gradients/default_policy/mul_6_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*
_output_shapes
: 
я
Cdefault_policy/gradients/default_policy/mul_6_grad/tuple/group_depsNoOp;^default_policy/gradients/default_policy/mul_6_grad/Reshape=^default_policy/gradients/default_policy/mul_6_grad/Reshape_1*&
 _has_manual_control_dependencies(
к
Kdefault_policy/gradients/default_policy/mul_6_grad/tuple/control_dependencyIdentity:default_policy/gradients/default_policy/mul_6_grad/ReshapeD^default_policy/gradients/default_policy/mul_6_grad/tuple/group_deps*
T0*M
_classC
A?loc:@default_policy/gradients/default_policy/mul_6_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Я
Mdefault_policy/gradients/default_policy/mul_6_grad/tuple/control_dependency_1Identity<default_policy/gradients/default_policy/mul_6_grad/Reshape_1D^default_policy/gradients/default_policy/mul_6_grad/tuple/group_deps*
T0*O
_classE
CAloc:@default_policy/gradients/default_policy/mul_6_grad/Reshape_1*
_output_shapes
: 

<default_policy/gradients/default_policy/truediv_6_grad/ShapeShapedefault_policy/add_9*
T0*
_output_shapes
:*
out_type0

>default_policy/gradients/default_policy/truediv_6_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
 
Ldefault_policy/gradients/default_policy/truediv_6_grad/BroadcastGradientArgsBroadcastGradientArgs<default_policy/gradients/default_policy/truediv_6_grad/Shape>default_policy/gradients/default_policy/truediv_6_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ф
>default_policy/gradients/default_policy/truediv_6_grad/RealDivRealDivKdefault_policy/gradients/default_policy/mul_6_grad/tuple/control_dependencydefault_policy/truediv_6/y*
T0*'
_output_shapes
:џџџџџџџџџ

:default_policy/gradients/default_policy/truediv_6_grad/SumSum>default_policy/gradients/default_policy/truediv_6_grad/RealDivLdefault_policy/gradients/default_policy/truediv_6_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Ћ
>default_policy/gradients/default_policy/truediv_6_grad/ReshapeReshape:default_policy/gradients/default_policy/truediv_6_grad/Sum<default_policy/gradients/default_policy/truediv_6_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ

:default_policy/gradients/default_policy/truediv_6_grad/NegNegdefault_policy/add_9*
T0*'
_output_shapes
:џџџџџџџџџ
е
@default_policy/gradients/default_policy/truediv_6_grad/RealDiv_1RealDiv:default_policy/gradients/default_policy/truediv_6_grad/Negdefault_policy/truediv_6/y*
T0*'
_output_shapes
:џџџџџџџџџ
л
@default_policy/gradients/default_policy/truediv_6_grad/RealDiv_2RealDiv@default_policy/gradients/default_policy/truediv_6_grad/RealDiv_1default_policy/truediv_6/y*
T0*'
_output_shapes
:џџџџџџџџџ

:default_policy/gradients/default_policy/truediv_6_grad/mulMulKdefault_policy/gradients/default_policy/mul_6_grad/tuple/control_dependency@default_policy/gradients/default_policy/truediv_6_grad/RealDiv_2*
T0*'
_output_shapes
:џџџџџџџџџ

<default_policy/gradients/default_policy/truediv_6_grad/Sum_1Sum:default_policy/gradients/default_policy/truediv_6_grad/mulNdefault_policy/gradients/default_policy/truediv_6_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
 
@default_policy/gradients/default_policy/truediv_6_grad/Reshape_1Reshape<default_policy/gradients/default_policy/truediv_6_grad/Sum_1>default_policy/gradients/default_policy/truediv_6_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*
_output_shapes
: 
ћ
Gdefault_policy/gradients/default_policy/truediv_6_grad/tuple/group_depsNoOp?^default_policy/gradients/default_policy/truediv_6_grad/ReshapeA^default_policy/gradients/default_policy/truediv_6_grad/Reshape_1*&
 _has_manual_control_dependencies(
ъ
Odefault_policy/gradients/default_policy/truediv_6_grad/tuple/control_dependencyIdentity>default_policy/gradients/default_policy/truediv_6_grad/ReshapeH^default_policy/gradients/default_policy/truediv_6_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@default_policy/gradients/default_policy/truediv_6_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
п
Qdefault_policy/gradients/default_policy/truediv_6_grad/tuple/control_dependency_1Identity@default_policy/gradients/default_policy/truediv_6_grad/Reshape_1H^default_policy/gradients/default_policy/truediv_6_grad/tuple/group_deps*
T0*S
_classI
GEloc:@default_policy/gradients/default_policy/truediv_6_grad/Reshape_1*
_output_shapes
: 

8default_policy/gradients/default_policy/add_9_grad/ShapeShapedefault_policy/Tanh_6*
T0*
_output_shapes
:*
out_type0

:default_policy/gradients/default_policy/add_9_grad/Shape_1Shapedefault_policy/add_9/y*
T0*
_output_shapes
: *
out_type0

Hdefault_policy/gradients/default_policy/add_9_grad/BroadcastGradientArgsBroadcastGradientArgs8default_policy/gradients/default_policy/add_9_grad/Shape:default_policy/gradients/default_policy/add_9_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

6default_policy/gradients/default_policy/add_9_grad/SumSumOdefault_policy/gradients/default_policy/truediv_6_grad/tuple/control_dependencyHdefault_policy/gradients/default_policy/add_9_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

:default_policy/gradients/default_policy/add_9_grad/ReshapeReshape6default_policy/gradients/default_policy/add_9_grad/Sum8default_policy/gradients/default_policy/add_9_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ

8default_policy/gradients/default_policy/add_9_grad/Sum_1SumOdefault_policy/gradients/default_policy/truediv_6_grad/tuple/control_dependencyJdefault_policy/gradients/default_policy/add_9_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

<default_policy/gradients/default_policy/add_9_grad/Reshape_1Reshape8default_policy/gradients/default_policy/add_9_grad/Sum_1:default_policy/gradients/default_policy/add_9_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*
_output_shapes
: 
я
Cdefault_policy/gradients/default_policy/add_9_grad/tuple/group_depsNoOp;^default_policy/gradients/default_policy/add_9_grad/Reshape=^default_policy/gradients/default_policy/add_9_grad/Reshape_1*&
 _has_manual_control_dependencies(

Kdefault_policy/gradients/default_policy/add_9_grad/tuple/control_dependencyIdentity:default_policy/gradients/default_policy/add_9_grad/ReshapeD^default_policy/gradients/default_policy/add_9_grad/tuple/group_deps*
T0*M
_classC
A?loc:@default_policy/gradients/default_policy/add_9_grad/Reshape*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
Я
Mdefault_policy/gradients/default_policy/add_9_grad/tuple/control_dependency_1Identity<default_policy/gradients/default_policy/add_9_grad/Reshape_1D^default_policy/gradients/default_policy/add_9_grad/tuple/group_deps*
T0*O
_classE
CAloc:@default_policy/gradients/default_policy/add_9_grad/Reshape_1*
_output_shapes
: 
о
<default_policy/gradients/default_policy/Tanh_6_grad/TanhGradTanhGraddefault_policy/Tanh_6Kdefault_policy/gradients/default_policy/add_9_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
Ю
[default_policy/gradients/default_policy/default_policy_Normal_2_1/sample/Reshape_grad/ShapeShape3default_policy/default_policy_Normal_2_1/sample/add*
T0*
_output_shapes
:*
out_type0
Ч
]default_policy/gradients/default_policy/default_policy_Normal_2_1/sample/Reshape_grad/ReshapeReshape<default_policy/gradients/default_policy/Tanh_6_grad/TanhGrad[default_policy/gradients/default_policy/default_policy_Normal_2_1/sample/Reshape_grad/Shape*
T0*
Tshape0*+
_output_shapes
:џџџџџџџџџ
Ъ
Wdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/ShapeShape3default_policy/default_policy_Normal_2_1/sample/mul*
T0*
_output_shapes
:*
out_type0
Џ
Ydefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/Shape_1Shapedefault_policy/split_2*
T0*
_output_shapes
:*
out_type0
ё
gdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/BroadcastGradientArgsBroadcastGradientArgsWdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/ShapeYdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ф
Udefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/SumSum]default_policy/gradients/default_policy/default_policy_Normal_2_1/sample/Reshape_grad/Reshapegdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

Ydefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/ReshapeReshapeUdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/SumWdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*+
_output_shapes
:џџџџџџџџџ
ш
Wdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/Sum_1Sum]default_policy/gradients/default_policy/default_policy_Normal_2_1/sample/Reshape_grad/Reshapeidefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

[default_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/Reshape_1ReshapeWdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/Sum_1Ydefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
Ь
bdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/tuple/group_depsNoOpZ^default_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/Reshape\^default_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/Reshape_1*&
 _has_manual_control_dependencies(
к
jdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/tuple/control_dependencyIdentityYdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/Reshapec^default_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/tuple/group_deps*
T0*l
_classb
`^loc:@default_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/Reshape*+
_output_shapes
:џџџџџџџџџ
м
ldefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/tuple/control_dependency_1Identity[default_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/Reshape_1c^default_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/tuple/group_deps*
T0*n
_classd
b`loc:@default_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
л
Wdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/ShapeShapeDdefault_policy/default_policy_Normal_2_1/sample/normal/random_normal*
T0*
_output_shapes
:*
out_type0
­
Ydefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/Shape_1Shapedefault_policy/Exp_5*
T0*
_output_shapes
:*
out_type0
ё
gdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/BroadcastGradientArgsBroadcastGradientArgsWdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/ShapeYdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Udefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/MulMuljdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/tuple/control_dependencydefault_policy/Exp_5*
T0*+
_output_shapes
:џџџџџџџџџ
м
Udefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/SumSumUdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/Mulgdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

Ydefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/ReshapeReshapeUdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/SumWdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
Ц
Wdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/Mul_1MulDdefault_policy/default_policy_Normal_2_1/sample/normal/random_normaljdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/tuple/control_dependency*
T0*+
_output_shapes
:џџџџџџџџџ
т
Wdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/Sum_1SumWdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/Mul_1idefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

[default_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/Reshape_1ReshapeWdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/Sum_1Ydefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
Ь
bdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/tuple/group_depsNoOpZ^default_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/Reshape\^default_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/Reshape_1*&
 _has_manual_control_dependencies(
у
jdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/tuple/control_dependencyIdentityYdefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/Reshapec^default_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/tuple/group_deps*
T0*l
_classb
`^loc:@default_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/Reshape*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
м
ldefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/tuple/control_dependency_1Identity[default_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/Reshape_1c^default_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/tuple/group_deps*
T0*n
_classd
b`loc:@default_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Ї
default_policy/gradients/AddN_2AddNWdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/Log_grad/mulrdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_grad/tuple/control_dependency_1tdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/tuple/control_dependency_1ldefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/mul_grad/tuple/control_dependency_1*
N*
T0*j
_class`
^\loc:@default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/Log_grad/mul*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
І
6default_policy/gradients/default_policy/Exp_5_grad/mulMuldefault_policy/gradients/AddN_2default_policy/Exp_5*
T0*'
_output_shapes
:џџџџџџџџџ
Њ
Cdefault_policy/gradients/default_policy/clip_by_value_11_grad/ShapeShape'default_policy/clip_by_value_11/Minimum*
T0*
_output_shapes
:*
out_type0

Edefault_policy/gradients/default_policy/clip_by_value_11_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
П
Hdefault_policy/gradients/default_policy/clip_by_value_11_grad/zeros_like	ZerosLike6default_policy/gradients/default_policy/Exp_5_grad/mul*
T0*'
_output_shapes
:џџџџџџџџџ
и
Jdefault_policy/gradients/default_policy/clip_by_value_11_grad/GreaterEqualGreaterEqual'default_policy/clip_by_value_11/Minimum!default_policy/clip_by_value_11/y*
T0*'
_output_shapes
:џџџџџџџџџ
Е
Sdefault_policy/gradients/default_policy/clip_by_value_11_grad/BroadcastGradientArgsBroadcastGradientArgsCdefault_policy/gradients/default_policy/clip_by_value_11_grad/ShapeEdefault_policy/gradients/default_policy/clip_by_value_11_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
в
Fdefault_policy/gradients/default_policy/clip_by_value_11_grad/SelectV2SelectV2Jdefault_policy/gradients/default_policy/clip_by_value_11_grad/GreaterEqual6default_policy/gradients/default_policy/Exp_5_grad/mulHdefault_policy/gradients/default_policy/clip_by_value_11_grad/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
Ѕ
Adefault_policy/gradients/default_policy/clip_by_value_11_grad/SumSumFdefault_policy/gradients/default_policy/clip_by_value_11_grad/SelectV2Sdefault_policy/gradients/default_policy/clip_by_value_11_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Р
Edefault_policy/gradients/default_policy/clip_by_value_11_grad/ReshapeReshapeAdefault_policy/gradients/default_policy/clip_by_value_11_grad/SumCdefault_policy/gradients/default_policy/clip_by_value_11_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
д
Hdefault_policy/gradients/default_policy/clip_by_value_11_grad/SelectV2_1SelectV2Jdefault_policy/gradients/default_policy/clip_by_value_11_grad/GreaterEqualHdefault_policy/gradients/default_policy/clip_by_value_11_grad/zeros_like6default_policy/gradients/default_policy/Exp_5_grad/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Ћ
Cdefault_policy/gradients/default_policy/clip_by_value_11_grad/Sum_1SumHdefault_policy/gradients/default_policy/clip_by_value_11_grad/SelectV2_1Udefault_policy/gradients/default_policy/clip_by_value_11_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Е
Gdefault_policy/gradients/default_policy/clip_by_value_11_grad/Reshape_1ReshapeCdefault_policy/gradients/default_policy/clip_by_value_11_grad/Sum_1Edefault_policy/gradients/default_policy/clip_by_value_11_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*
_output_shapes
: 

Ndefault_policy/gradients/default_policy/clip_by_value_11_grad/tuple/group_depsNoOpF^default_policy/gradients/default_policy/clip_by_value_11_grad/ReshapeH^default_policy/gradients/default_policy/clip_by_value_11_grad/Reshape_1*&
 _has_manual_control_dependencies(

Vdefault_policy/gradients/default_policy/clip_by_value_11_grad/tuple/control_dependencyIdentityEdefault_policy/gradients/default_policy/clip_by_value_11_grad/ReshapeO^default_policy/gradients/default_policy/clip_by_value_11_grad/tuple/group_deps*
T0*X
_classN
LJloc:@default_policy/gradients/default_policy/clip_by_value_11_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
ћ
Xdefault_policy/gradients/default_policy/clip_by_value_11_grad/tuple/control_dependency_1IdentityGdefault_policy/gradients/default_policy/clip_by_value_11_grad/Reshape_1O^default_policy/gradients/default_policy/clip_by_value_11_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@default_policy/gradients/default_policy/clip_by_value_11_grad/Reshape_1*
_output_shapes
: 
Ѓ
Kdefault_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/ShapeShapedefault_policy/split_2:1*
T0*
_output_shapes
:*
out_type0

Mdefault_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
ч
Pdefault_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/zeros_like	ZerosLikeVdefault_policy/gradients/default_policy/clip_by_value_11_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
г
Odefault_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/LessEqual	LessEqualdefault_policy/split_2:1)default_policy/clip_by_value_11/Minimum/y*
T0*'
_output_shapes
:џџџџџџџџџ
Э
[default_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsKdefault_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/ShapeMdefault_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Ndefault_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/SelectV2SelectV2Odefault_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/LessEqualVdefault_policy/gradients/default_policy/clip_by_value_11_grad/tuple/control_dependencyPdefault_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
Н
Idefault_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/SumSumNdefault_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/SelectV2[default_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
и
Mdefault_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/ReshapeReshapeIdefault_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/SumKdefault_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ

Pdefault_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/SelectV2_1SelectV2Odefault_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/LessEqualPdefault_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/zeros_likeVdefault_policy/gradients/default_policy/clip_by_value_11_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
У
Kdefault_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/Sum_1SumPdefault_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/SelectV2_1]default_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Э
Odefault_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/Reshape_1ReshapeKdefault_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/Sum_1Mdefault_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*
_output_shapes
: 
Ј
Vdefault_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/tuple/group_depsNoOpN^default_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/ReshapeP^default_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/Reshape_1*&
 _has_manual_control_dependencies(
І
^default_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/tuple/control_dependencyIdentityMdefault_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/ReshapeW^default_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/tuple/group_deps*
T0*`
_classV
TRloc:@default_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

`default_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/tuple/control_dependency_1IdentityOdefault_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/Reshape_1W^default_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/tuple/group_deps*
T0*b
_classX
VTloc:@default_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/Reshape_1*
_output_shapes
: 
К
default_policy/gradients/AddN_3AddNrdefault_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/tuple/control_dependencyldefault_policy/gradients/default_policy/default_policy_Normal_2_1/sample/add_grad/tuple/control_dependency_1*
N*
T0*t
_classj
hfloc:@default_policy/gradients/default_policy/default_policy_Normal_2_3/log_prob/truediv_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
й
;default_policy/gradients/default_policy/split_2_grad/concatConcatV2default_policy/gradients/AddN_3^default_policy/gradients/default_policy/clip_by_value_11/Minimum_grad/tuple/control_dependency default_policy/split_2/split_dim*
N*
T0*

Tidx0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ

Xdefault_policy/gradients/default_policy/sequential_7/action_out/BiasAdd_grad/BiasAddGradBiasAddGrad;default_policy/gradients/default_policy/split_2_grad/concat*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:*
data_formatNHWC
І
]default_policy/gradients/default_policy/sequential_7/action_out/BiasAdd_grad/tuple/group_depsNoOpY^default_policy/gradients/default_policy/sequential_7/action_out/BiasAdd_grad/BiasAddGrad<^default_policy/gradients/default_policy/split_2_grad/concat*&
 _has_manual_control_dependencies(

edefault_policy/gradients/default_policy/sequential_7/action_out/BiasAdd_grad/tuple/control_dependencyIdentity;default_policy/gradients/default_policy/split_2_grad/concat^^default_policy/gradients/default_policy/sequential_7/action_out/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@default_policy/gradients/default_policy/split_2_grad/concat*'
_output_shapes
:џџџџџџџџџ
П
gdefault_policy/gradients/default_policy/sequential_7/action_out/BiasAdd_grad/tuple/control_dependency_1IdentityXdefault_policy/gradients/default_policy/sequential_7/action_out/BiasAdd_grad/BiasAddGrad^^default_policy/gradients/default_policy/sequential_7/action_out/BiasAdd_grad/tuple/group_deps*
T0*k
_classa
_]loc:@default_policy/gradients/default_policy/sequential_7/action_out/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

Rdefault_policy/gradients/default_policy/sequential_7/action_out/MatMul_grad/MatMulMatMuledefault_policy/gradients/default_policy/sequential_7/action_out/BiasAdd_grad/tuple/control_dependency<default_policy/sequential_7/action_out/MatMul/ReadVariableOp*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
ш
Tdefault_policy/gradients/default_policy/sequential_7/action_out/MatMul_grad/MatMul_1MatMul)default_policy/sequential_7/action_2/Reluedefault_policy/gradients/default_policy/sequential_7/action_out/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:	*
transpose_a(*
transpose_b( 
И
\default_policy/gradients/default_policy/sequential_7/action_out/MatMul_grad/tuple/group_depsNoOpS^default_policy/gradients/default_policy/sequential_7/action_out/MatMul_grad/MatMulU^default_policy/gradients/default_policy/sequential_7/action_out/MatMul_grad/MatMul_1*&
 _has_manual_control_dependencies(
Н
ddefault_policy/gradients/default_policy/sequential_7/action_out/MatMul_grad/tuple/control_dependencyIdentityRdefault_policy/gradients/default_policy/sequential_7/action_out/MatMul_grad/MatMul]^default_policy/gradients/default_policy/sequential_7/action_out/MatMul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@default_policy/gradients/default_policy/sequential_7/action_out/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
К
fdefault_policy/gradients/default_policy/sequential_7/action_out/MatMul_grad/tuple/control_dependency_1IdentityTdefault_policy/gradients/default_policy/sequential_7/action_out/MatMul_grad/MatMul_1]^default_policy/gradients/default_policy/sequential_7/action_out/MatMul_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@default_policy/gradients/default_policy/sequential_7/action_out/MatMul_grad/MatMul_1*
_output_shapes
:	
Ш
Pdefault_policy/gradients/default_policy/sequential_7/action_2/Relu_grad/ReluGradReluGradddefault_policy/gradients/default_policy/sequential_7/action_out/MatMul_grad/tuple/control_dependency)default_policy/sequential_7/action_2/Relu*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:џџџџџџџџџ

Vdefault_policy/gradients/default_policy/sequential_7/action_2/BiasAdd_grad/BiasAddGradBiasAddGradPdefault_policy/gradients/default_policy/sequential_7/action_2/Relu_grad/ReluGrad*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:*
data_formatNHWC
З
[default_policy/gradients/default_policy/sequential_7/action_2/BiasAdd_grad/tuple/group_depsNoOpW^default_policy/gradients/default_policy/sequential_7/action_2/BiasAdd_grad/BiasAddGradQ^default_policy/gradients/default_policy/sequential_7/action_2/Relu_grad/ReluGrad*&
 _has_manual_control_dependencies(
З
cdefault_policy/gradients/default_policy/sequential_7/action_2/BiasAdd_grad/tuple/control_dependencyIdentityPdefault_policy/gradients/default_policy/sequential_7/action_2/Relu_grad/ReluGrad\^default_policy/gradients/default_policy/sequential_7/action_2/BiasAdd_grad/tuple/group_deps*
T0*c
_classY
WUloc:@default_policy/gradients/default_policy/sequential_7/action_2/Relu_grad/ReluGrad*(
_output_shapes
:џџџџџџџџџ
И
edefault_policy/gradients/default_policy/sequential_7/action_2/BiasAdd_grad/tuple/control_dependency_1IdentityVdefault_policy/gradients/default_policy/sequential_7/action_2/BiasAdd_grad/BiasAddGrad\^default_policy/gradients/default_policy/sequential_7/action_2/BiasAdd_grad/tuple/group_deps*
T0*i
_class_
][loc:@default_policy/gradients/default_policy/sequential_7/action_2/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
ќ
Pdefault_policy/gradients/default_policy/sequential_7/action_2/MatMul_grad/MatMulMatMulcdefault_policy/gradients/default_policy/sequential_7/action_2/BiasAdd_grad/tuple/control_dependency:default_policy/sequential_7/action_2/MatMul/ReadVariableOp*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
х
Rdefault_policy/gradients/default_policy/sequential_7/action_2/MatMul_grad/MatMul_1MatMul)default_policy/sequential_7/action_1/Relucdefault_policy/gradients/default_policy/sequential_7/action_2/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
В
Zdefault_policy/gradients/default_policy/sequential_7/action_2/MatMul_grad/tuple/group_depsNoOpQ^default_policy/gradients/default_policy/sequential_7/action_2/MatMul_grad/MatMulS^default_policy/gradients/default_policy/sequential_7/action_2/MatMul_grad/MatMul_1*&
 _has_manual_control_dependencies(
Е
bdefault_policy/gradients/default_policy/sequential_7/action_2/MatMul_grad/tuple/control_dependencyIdentityPdefault_policy/gradients/default_policy/sequential_7/action_2/MatMul_grad/MatMul[^default_policy/gradients/default_policy/sequential_7/action_2/MatMul_grad/tuple/group_deps*
T0*c
_classY
WUloc:@default_policy/gradients/default_policy/sequential_7/action_2/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
Г
ddefault_policy/gradients/default_policy/sequential_7/action_2/MatMul_grad/tuple/control_dependency_1IdentityRdefault_policy/gradients/default_policy/sequential_7/action_2/MatMul_grad/MatMul_1[^default_policy/gradients/default_policy/sequential_7/action_2/MatMul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@default_policy/gradients/default_policy/sequential_7/action_2/MatMul_grad/MatMul_1* 
_output_shapes
:

Ц
Pdefault_policy/gradients/default_policy/sequential_7/action_1/Relu_grad/ReluGradReluGradbdefault_policy/gradients/default_policy/sequential_7/action_2/MatMul_grad/tuple/control_dependency)default_policy/sequential_7/action_1/Relu*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:џџџџџџџџџ

Vdefault_policy/gradients/default_policy/sequential_7/action_1/BiasAdd_grad/BiasAddGradBiasAddGradPdefault_policy/gradients/default_policy/sequential_7/action_1/Relu_grad/ReluGrad*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:*
data_formatNHWC
З
[default_policy/gradients/default_policy/sequential_7/action_1/BiasAdd_grad/tuple/group_depsNoOpW^default_policy/gradients/default_policy/sequential_7/action_1/BiasAdd_grad/BiasAddGradQ^default_policy/gradients/default_policy/sequential_7/action_1/Relu_grad/ReluGrad*&
 _has_manual_control_dependencies(
З
cdefault_policy/gradients/default_policy/sequential_7/action_1/BiasAdd_grad/tuple/control_dependencyIdentityPdefault_policy/gradients/default_policy/sequential_7/action_1/Relu_grad/ReluGrad\^default_policy/gradients/default_policy/sequential_7/action_1/BiasAdd_grad/tuple/group_deps*
T0*c
_classY
WUloc:@default_policy/gradients/default_policy/sequential_7/action_1/Relu_grad/ReluGrad*(
_output_shapes
:џџџџџџџџџ
И
edefault_policy/gradients/default_policy/sequential_7/action_1/BiasAdd_grad/tuple/control_dependency_1IdentityVdefault_policy/gradients/default_policy/sequential_7/action_1/BiasAdd_grad/BiasAddGrad\^default_policy/gradients/default_policy/sequential_7/action_1/BiasAdd_grad/tuple/group_deps*
T0*i
_class_
][loc:@default_policy/gradients/default_policy/sequential_7/action_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
ќ
Pdefault_policy/gradients/default_policy/sequential_7/action_1/MatMul_grad/MatMulMatMulcdefault_policy/gradients/default_policy/sequential_7/action_1/BiasAdd_grad/tuple/control_dependency:default_policy/sequential_7/action_1/MatMul/ReadVariableOp*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
ж
Rdefault_policy/gradients/default_policy/sequential_7/action_1/MatMul_grad/MatMul_1MatMuldefault_policy/observationcdefault_policy/gradients/default_policy/sequential_7/action_1/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
В
Zdefault_policy/gradients/default_policy/sequential_7/action_1/MatMul_grad/tuple/group_depsNoOpQ^default_policy/gradients/default_policy/sequential_7/action_1/MatMul_grad/MatMulS^default_policy/gradients/default_policy/sequential_7/action_1/MatMul_grad/MatMul_1*&
 _has_manual_control_dependencies(
Е
bdefault_policy/gradients/default_policy/sequential_7/action_1/MatMul_grad/tuple/control_dependencyIdentityPdefault_policy/gradients/default_policy/sequential_7/action_1/MatMul_grad/MatMul[^default_policy/gradients/default_policy/sequential_7/action_1/MatMul_grad/tuple/group_deps*
T0*c
_classY
WUloc:@default_policy/gradients/default_policy/sequential_7/action_1/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
Г
ddefault_policy/gradients/default_policy/sequential_7/action_1/MatMul_grad/tuple/control_dependency_1IdentityRdefault_policy/gradients/default_policy/sequential_7/action_1/MatMul_grad/MatMul_1[^default_policy/gradients/default_policy/sequential_7/action_1/MatMul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@default_policy/gradients/default_policy/sequential_7/action_1/MatMul_grad/MatMul_1* 
_output_shapes
:

c
 default_policy/gradients_1/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
o
*default_policy/gradients_1/grad_ys_0/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
­
$default_policy/gradients_1/grad_ys_0Fill default_policy/gradients_1/Shape*default_policy/gradients_1/grad_ys_0/Const*
T0*
_output_shapes
: *

index_type0
Ф
9default_policy/gradients_1/default_policy/mul_16_grad/MulMul$default_policy/gradients_1/grad_ys_0default_policy/Mean*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
: 
Ъ
;default_policy/gradients_1/default_policy/mul_16_grad/Mul_1Mul$default_policy/gradients_1/grad_ys_0default_policy/mul_16/x*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
: 
№
Fdefault_policy/gradients_1/default_policy/mul_16_grad/tuple/group_depsNoOp:^default_policy/gradients_1/default_policy/mul_16_grad/Mul<^default_policy/gradients_1/default_policy/mul_16_grad/Mul_1*&
 _has_manual_control_dependencies(
Э
Ndefault_policy/gradients_1/default_policy/mul_16_grad/tuple/control_dependencyIdentity9default_policy/gradients_1/default_policy/mul_16_grad/MulG^default_policy/gradients_1/default_policy/mul_16_grad/tuple/group_deps*
T0*L
_classB
@>loc:@default_policy/gradients_1/default_policy/mul_16_grad/Mul*
_output_shapes
: 
г
Pdefault_policy/gradients_1/default_policy/mul_16_grad/tuple/control_dependency_1Identity;default_policy/gradients_1/default_policy/mul_16_grad/Mul_1G^default_policy/gradients_1/default_policy/mul_16_grad/tuple/group_deps*
T0*N
_classD
B@loc:@default_policy/gradients_1/default_policy/mul_16_grad/Mul_1*
_output_shapes
: 

9default_policy/gradients_1/default_policy/Mean_grad/ShapeShape default_policy/SquaredDifference*
T0*
_output_shapes
:*
out_type0
Ш
8default_policy/gradients_1/default_policy/Mean_grad/SizeConst*L
_classB
@>loc:@default_policy/gradients_1/default_policy/Mean_grad/Shape*
_output_shapes
: *
dtype0*
value	B :

7default_policy/gradients_1/default_policy/Mean_grad/addAddV2%default_policy/Mean/reduction_indices8default_policy/gradients_1/default_policy/Mean_grad/Size*
T0*L
_classB
@>loc:@default_policy/gradients_1/default_policy/Mean_grad/Shape*
_output_shapes
: 
Ѕ
7default_policy/gradients_1/default_policy/Mean_grad/modFloorMod7default_policy/gradients_1/default_policy/Mean_grad/add8default_policy/gradients_1/default_policy/Mean_grad/Size*
T0*L
_classB
@>loc:@default_policy/gradients_1/default_policy/Mean_grad/Shape*
_output_shapes
: 
Ь
;default_policy/gradients_1/default_policy/Mean_grad/Shape_1Const*L
_classB
@>loc:@default_policy/gradients_1/default_policy/Mean_grad/Shape*
_output_shapes
: *
dtype0*
valueB 
Я
?default_policy/gradients_1/default_policy/Mean_grad/range/startConst*L
_classB
@>loc:@default_policy/gradients_1/default_policy/Mean_grad/Shape*
_output_shapes
: *
dtype0*
value	B : 
Я
?default_policy/gradients_1/default_policy/Mean_grad/range/deltaConst*L
_classB
@>loc:@default_policy/gradients_1/default_policy/Mean_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
є
9default_policy/gradients_1/default_policy/Mean_grad/rangeRange?default_policy/gradients_1/default_policy/Mean_grad/range/start8default_policy/gradients_1/default_policy/Mean_grad/Size?default_policy/gradients_1/default_policy/Mean_grad/range/delta*

Tidx0*L
_classB
@>loc:@default_policy/gradients_1/default_policy/Mean_grad/Shape*
_output_shapes
:
Ю
>default_policy/gradients_1/default_policy/Mean_grad/ones/ConstConst*L
_classB
@>loc:@default_policy/gradients_1/default_policy/Mean_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
О
8default_policy/gradients_1/default_policy/Mean_grad/onesFill;default_policy/gradients_1/default_policy/Mean_grad/Shape_1>default_policy/gradients_1/default_policy/Mean_grad/ones/Const*
T0*L
_classB
@>loc:@default_policy/gradients_1/default_policy/Mean_grad/Shape*
_output_shapes
: *

index_type0
З
Adefault_policy/gradients_1/default_policy/Mean_grad/DynamicStitchDynamicStitch9default_policy/gradients_1/default_policy/Mean_grad/range7default_policy/gradients_1/default_policy/Mean_grad/mod9default_policy/gradients_1/default_policy/Mean_grad/Shape8default_policy/gradients_1/default_policy/Mean_grad/ones*
N*
T0*L
_classB
@>loc:@default_policy/gradients_1/default_policy/Mean_grad/Shape*
_output_shapes
:

;default_policy/gradients_1/default_policy/Mean_grad/ReshapeReshapePdefault_policy/gradients_1/default_policy/mul_16_grad/tuple/control_dependency_1Adefault_policy/gradients_1/default_policy/Mean_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:

?default_policy/gradients_1/default_policy/Mean_grad/BroadcastToBroadcastTo;default_policy/gradients_1/default_policy/Mean_grad/Reshape9default_policy/gradients_1/default_policy/Mean_grad/Shape*
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ

;default_policy/gradients_1/default_policy/Mean_grad/Shape_2Shape default_policy/SquaredDifference*
T0*
_output_shapes
:*
out_type0
~
;default_policy/gradients_1/default_policy/Mean_grad/Shape_3Const*
_output_shapes
: *
dtype0*
valueB 

9default_policy/gradients_1/default_policy/Mean_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
і
8default_policy/gradients_1/default_policy/Mean_grad/ProdProd;default_policy/gradients_1/default_policy/Mean_grad/Shape_29default_policy/gradients_1/default_policy/Mean_grad/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 

;default_policy/gradients_1/default_policy/Mean_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
њ
:default_policy/gradients_1/default_policy/Mean_grad/Prod_1Prod;default_policy/gradients_1/default_policy/Mean_grad/Shape_3;default_policy/gradients_1/default_policy/Mean_grad/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 

=default_policy/gradients_1/default_policy/Mean_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :
т
;default_policy/gradients_1/default_policy/Mean_grad/MaximumMaximum:default_policy/gradients_1/default_policy/Mean_grad/Prod_1=default_policy/gradients_1/default_policy/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
р
<default_policy/gradients_1/default_policy/Mean_grad/floordivFloorDiv8default_policy/gradients_1/default_policy/Mean_grad/Prod;default_policy/gradients_1/default_policy/Mean_grad/Maximum*
T0*
_output_shapes
: 
О
8default_policy/gradients_1/default_policy/Mean_grad/CastCast<default_policy/gradients_1/default_policy/Mean_grad/floordiv*

DstT0*

SrcT0*
Truncate( *
_output_shapes
: 

;default_policy/gradients_1/default_policy/Mean_grad/truedivRealDiv?default_policy/gradients_1/default_policy/Mean_grad/BroadcastTo8default_policy/gradients_1/default_policy/Mean_grad/Cast*
T0*&
 _has_manual_control_dependencies(*#
_output_shapes
:џџџџџџџџџ
Ъ
Gdefault_policy/gradients_1/default_policy/SquaredDifference_grad/scalarConst<^default_policy/gradients_1/default_policy/Mean_grad/truediv*
_output_shapes
: *
dtype0*
valueB
 *   @
џ
Ddefault_policy/gradients_1/default_policy/SquaredDifference_grad/MulMulGdefault_policy/gradients_1/default_policy/SquaredDifference_grad/scalar;default_policy/gradients_1/default_policy/Mean_grad/truediv*
T0*#
_output_shapes
:џџџџџџџџџ
ь
Ddefault_policy/gradients_1/default_policy/SquaredDifference_grad/subSubdefault_policy/Squeezedefault_policy/StopGradient<^default_policy/gradients_1/default_policy/Mean_grad/truediv*
T0*#
_output_shapes
:џџџџџџџџџ

Fdefault_policy/gradients_1/default_policy/SquaredDifference_grad/mul_1MulDdefault_policy/gradients_1/default_policy/SquaredDifference_grad/MulDdefault_policy/gradients_1/default_policy/SquaredDifference_grad/sub*
T0*#
_output_shapes
:џџџџџџџџџ

Fdefault_policy/gradients_1/default_policy/SquaredDifference_grad/ShapeShapedefault_policy/Squeeze*
T0*
_output_shapes
:*
out_type0
Ѓ
Hdefault_policy/gradients_1/default_policy/SquaredDifference_grad/Shape_1Shapedefault_policy/StopGradient*
T0*
_output_shapes
:*
out_type0
О
Vdefault_policy/gradients_1/default_policy/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsFdefault_policy/gradients_1/default_policy/SquaredDifference_grad/ShapeHdefault_policy/gradients_1/default_policy/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
Ddefault_policy/gradients_1/default_policy/SquaredDifference_grad/SumSumFdefault_policy/gradients_1/default_policy/SquaredDifference_grad/mul_1Vdefault_policy/gradients_1/default_policy/SquaredDifference_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Х
Hdefault_policy/gradients_1/default_policy/SquaredDifference_grad/ReshapeReshapeDdefault_policy/gradients_1/default_policy/SquaredDifference_grad/SumFdefault_policy/gradients_1/default_policy/SquaredDifference_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*#
_output_shapes
:џџџџџџџџџ
Џ
Fdefault_policy/gradients_1/default_policy/SquaredDifference_grad/Sum_1SumFdefault_policy/gradients_1/default_policy/SquaredDifference_grad/mul_1Xdefault_policy/gradients_1/default_policy/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Ѓ
Jdefault_policy/gradients_1/default_policy/SquaredDifference_grad/Reshape_1ReshapeFdefault_policy/gradients_1/default_policy/SquaredDifference_grad/Sum_1Hdefault_policy/gradients_1/default_policy/SquaredDifference_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ
э
Ddefault_policy/gradients_1/default_policy/SquaredDifference_grad/NegNegJdefault_policy/gradients_1/default_policy/SquaredDifference_grad/Reshape_1*
T0*&
 _has_manual_control_dependencies(*#
_output_shapes
:џџџџџџџџџ

Qdefault_policy/gradients_1/default_policy/SquaredDifference_grad/tuple/group_depsNoOpE^default_policy/gradients_1/default_policy/SquaredDifference_grad/NegI^default_policy/gradients_1/default_policy/SquaredDifference_grad/Reshape*&
 _has_manual_control_dependencies(

Ydefault_policy/gradients_1/default_policy/SquaredDifference_grad/tuple/control_dependencyIdentityHdefault_policy/gradients_1/default_policy/SquaredDifference_grad/ReshapeR^default_policy/gradients_1/default_policy/SquaredDifference_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@default_policy/gradients_1/default_policy/SquaredDifference_grad/Reshape*#
_output_shapes
:џџџџџџџџџ

[default_policy/gradients_1/default_policy/SquaredDifference_grad/tuple/control_dependency_1IdentityDdefault_policy/gradients_1/default_policy/SquaredDifference_grad/NegR^default_policy/gradients_1/default_policy/SquaredDifference_grad/tuple/group_deps*
T0*W
_classM
KIloc:@default_policy/gradients_1/default_policy/SquaredDifference_grad/Neg*#
_output_shapes
:џџџџџџџџџ
Џ
<default_policy/gradients_1/default_policy/Squeeze_grad/ShapeShape3default_policy/model_1_1/sequential_1/q_out/BiasAdd*
T0*
_output_shapes
:*
out_type0
Ъ
>default_policy/gradients_1/default_policy/Squeeze_grad/ReshapeReshapeYdefault_policy/gradients_1/default_policy/SquaredDifference_grad/tuple/control_dependency<default_policy/gradients_1/default_policy/Squeeze_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ

_default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_out/BiasAdd_grad/BiasAddGradBiasAddGrad>default_policy/gradients_1/default_policy/Squeeze_grad/Reshape*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:*
data_formatNHWC
З
ddefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_out/BiasAdd_grad/tuple/group_depsNoOp?^default_policy/gradients_1/default_policy/Squeeze_grad/Reshape`^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_out/BiasAdd_grad/BiasAddGrad*&
 _has_manual_control_dependencies(
Є
ldefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_out/BiasAdd_grad/tuple/control_dependencyIdentity>default_policy/gradients_1/default_policy/Squeeze_grad/Reshapee^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_out/BiasAdd_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@default_policy/gradients_1/default_policy/Squeeze_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
л
ndefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_out/BiasAdd_grad/tuple/control_dependency_1Identity_default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_out/BiasAdd_grad/BiasAddGrade^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_out/BiasAdd_grad/tuple/group_deps*
T0*r
_classh
fdloc:@default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_out/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

Ydefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_out/MatMul_grad/MatMulMatMulldefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_out/BiasAdd_grad/tuple/control_dependencyAdefault_policy/model_1_1/sequential_1/q_out/MatMul/ReadVariableOp*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(

[default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_out/MatMul_grad/MatMul_1MatMul5default_policy/model_1_1/sequential_1/q_hidden_1/Reluldefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_out/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:	*
transpose_a(*
transpose_b( 
Э
cdefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_out/MatMul_grad/tuple/group_depsNoOpZ^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_out/MatMul_grad/MatMul\^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_out/MatMul_grad/MatMul_1*&
 _has_manual_control_dependencies(
й
kdefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_out/MatMul_grad/tuple/control_dependencyIdentityYdefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_out/MatMul_grad/MatMuld^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_out/MatMul_grad/tuple/group_deps*
T0*l
_classb
`^loc:@default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_out/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
ж
mdefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_out/MatMul_grad/tuple/control_dependency_1Identity[default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_out/MatMul_grad/MatMul_1d^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_out/MatMul_grad/tuple/group_deps*
T0*n
_classd
b`loc:@default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_out/MatMul_grad/MatMul_1*
_output_shapes
:	
щ
^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/Relu_grad/ReluGradReluGradkdefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_out/MatMul_grad/tuple/control_dependency5default_policy/model_1_1/sequential_1/q_hidden_1/Relu*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:џџџџџџџџџ
И
ddefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/BiasAdd_grad/BiasAddGradBiasAddGrad^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/Relu_grad/ReluGrad*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:*
data_formatNHWC
с
idefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/BiasAdd_grad/tuple/group_depsNoOpe^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/BiasAdd_grad/BiasAddGrad_^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/Relu_grad/ReluGrad*&
 _has_manual_control_dependencies(
я
qdefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/BiasAdd_grad/tuple/control_dependencyIdentity^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/Relu_grad/ReluGradj^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/BiasAdd_grad/tuple/group_deps*
T0*q
_classg
ecloc:@default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/Relu_grad/ReluGrad*(
_output_shapes
:џџџџџџџџџ
№
sdefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/BiasAdd_grad/tuple/control_dependency_1Identityddefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/BiasAdd_grad/BiasAddGradj^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/BiasAdd_grad/tuple/group_deps*
T0*w
_classm
kiloc:@default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Є
^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/MatMul_grad/MatMulMatMulqdefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/BiasAdd_grad/tuple/control_dependencyFdefault_policy/model_1_1/sequential_1/q_hidden_1/MatMul/ReadVariableOp*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(

`default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/MatMul_grad/MatMul_1MatMul5default_policy/model_1_1/sequential_1/q_hidden_0/Reluqdefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
м
hdefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/MatMul_grad/tuple/group_depsNoOp_^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/MatMul_grad/MatMula^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/MatMul_grad/MatMul_1*&
 _has_manual_control_dependencies(
э
pdefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/MatMul_grad/tuple/control_dependencyIdentity^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/MatMul_grad/MatMuli^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/MatMul_grad/tuple/group_deps*
T0*q
_classg
ecloc:@default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
ы
rdefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/MatMul_grad/tuple/control_dependency_1Identity`default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/MatMul_grad/MatMul_1i^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/MatMul_grad/tuple/group_deps*
T0*s
_classi
geloc:@default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/MatMul_grad/MatMul_1* 
_output_shapes
:

ю
^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/Relu_grad/ReluGradReluGradpdefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/MatMul_grad/tuple/control_dependency5default_policy/model_1_1/sequential_1/q_hidden_0/Relu*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:џџџџџџџџџ
И
ddefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/BiasAdd_grad/BiasAddGradBiasAddGrad^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/Relu_grad/ReluGrad*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:*
data_formatNHWC
с
idefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/BiasAdd_grad/tuple/group_depsNoOpe^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/BiasAdd_grad/BiasAddGrad_^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/Relu_grad/ReluGrad*&
 _has_manual_control_dependencies(
я
qdefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/BiasAdd_grad/tuple/control_dependencyIdentity^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/Relu_grad/ReluGradj^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/BiasAdd_grad/tuple/group_deps*
T0*q
_classg
ecloc:@default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/Relu_grad/ReluGrad*(
_output_shapes
:џџџџџџџџџ
№
sdefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/BiasAdd_grad/tuple/control_dependency_1Identityddefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/BiasAdd_grad/BiasAddGradj^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/BiasAdd_grad/tuple/group_deps*
T0*w
_classm
kiloc:@default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Є
^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/MatMul_grad/MatMulMatMulqdefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/BiasAdd_grad/tuple/control_dependencyFdefault_policy/model_1_1/sequential_1/q_hidden_0/MatMul/ReadVariableOp*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(

`default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/MatMul_grad/MatMul_1MatMul8default_policy/model_1_1/sequential_1/concatenate/concatqdefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
м
hdefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/MatMul_grad/tuple/group_depsNoOp_^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/MatMul_grad/MatMula^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/MatMul_grad/MatMul_1*&
 _has_manual_control_dependencies(
э
pdefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/MatMul_grad/tuple/control_dependencyIdentity^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/MatMul_grad/MatMuli^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/MatMul_grad/tuple/group_deps*
T0*q
_classg
ecloc:@default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
ы
rdefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/MatMul_grad/tuple/control_dependency_1Identity`default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/MatMul_grad/MatMul_1i^default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/MatMul_grad/tuple/group_deps*
T0*s
_classi
geloc:@default_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/MatMul_grad/MatMul_1* 
_output_shapes
:

c
 default_policy/gradients_2/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
o
*default_policy/gradients_2/grad_ys_0/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
­
$default_policy/gradients_2/grad_ys_0Fill default_policy/gradients_2/Shape*default_policy/gradients_2/grad_ys_0/Const*
T0*
_output_shapes
: *

index_type0
Ц
9default_policy/gradients_2/default_policy/mul_17_grad/MulMul$default_policy/gradients_2/grad_ys_0default_policy/Mean_1*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
: 
Ъ
;default_policy/gradients_2/default_policy/mul_17_grad/Mul_1Mul$default_policy/gradients_2/grad_ys_0default_policy/mul_17/x*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
: 
№
Fdefault_policy/gradients_2/default_policy/mul_17_grad/tuple/group_depsNoOp:^default_policy/gradients_2/default_policy/mul_17_grad/Mul<^default_policy/gradients_2/default_policy/mul_17_grad/Mul_1*&
 _has_manual_control_dependencies(
Э
Ndefault_policy/gradients_2/default_policy/mul_17_grad/tuple/control_dependencyIdentity9default_policy/gradients_2/default_policy/mul_17_grad/MulG^default_policy/gradients_2/default_policy/mul_17_grad/tuple/group_deps*
T0*L
_classB
@>loc:@default_policy/gradients_2/default_policy/mul_17_grad/Mul*
_output_shapes
: 
г
Pdefault_policy/gradients_2/default_policy/mul_17_grad/tuple/control_dependency_1Identity;default_policy/gradients_2/default_policy/mul_17_grad/Mul_1G^default_policy/gradients_2/default_policy/mul_17_grad/tuple/group_deps*
T0*N
_classD
B@loc:@default_policy/gradients_2/default_policy/mul_17_grad/Mul_1*
_output_shapes
: 

;default_policy/gradients_2/default_policy/Mean_1_grad/ShapeShape"default_policy/SquaredDifference_1*
T0*
_output_shapes
:*
out_type0
Ь
:default_policy/gradients_2/default_policy/Mean_1_grad/SizeConst*N
_classD
B@loc:@default_policy/gradients_2/default_policy/Mean_1_grad/Shape*
_output_shapes
: *
dtype0*
value	B :

9default_policy/gradients_2/default_policy/Mean_1_grad/addAddV2'default_policy/Mean_1/reduction_indices:default_policy/gradients_2/default_policy/Mean_1_grad/Size*
T0*N
_classD
B@loc:@default_policy/gradients_2/default_policy/Mean_1_grad/Shape*
_output_shapes
: 
­
9default_policy/gradients_2/default_policy/Mean_1_grad/modFloorMod9default_policy/gradients_2/default_policy/Mean_1_grad/add:default_policy/gradients_2/default_policy/Mean_1_grad/Size*
T0*N
_classD
B@loc:@default_policy/gradients_2/default_policy/Mean_1_grad/Shape*
_output_shapes
: 
а
=default_policy/gradients_2/default_policy/Mean_1_grad/Shape_1Const*N
_classD
B@loc:@default_policy/gradients_2/default_policy/Mean_1_grad/Shape*
_output_shapes
: *
dtype0*
valueB 
г
Adefault_policy/gradients_2/default_policy/Mean_1_grad/range/startConst*N
_classD
B@loc:@default_policy/gradients_2/default_policy/Mean_1_grad/Shape*
_output_shapes
: *
dtype0*
value	B : 
г
Adefault_policy/gradients_2/default_policy/Mean_1_grad/range/deltaConst*N
_classD
B@loc:@default_policy/gradients_2/default_policy/Mean_1_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
ў
;default_policy/gradients_2/default_policy/Mean_1_grad/rangeRangeAdefault_policy/gradients_2/default_policy/Mean_1_grad/range/start:default_policy/gradients_2/default_policy/Mean_1_grad/SizeAdefault_policy/gradients_2/default_policy/Mean_1_grad/range/delta*

Tidx0*N
_classD
B@loc:@default_policy/gradients_2/default_policy/Mean_1_grad/Shape*
_output_shapes
:
в
@default_policy/gradients_2/default_policy/Mean_1_grad/ones/ConstConst*N
_classD
B@loc:@default_policy/gradients_2/default_policy/Mean_1_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
Ц
:default_policy/gradients_2/default_policy/Mean_1_grad/onesFill=default_policy/gradients_2/default_policy/Mean_1_grad/Shape_1@default_policy/gradients_2/default_policy/Mean_1_grad/ones/Const*
T0*N
_classD
B@loc:@default_policy/gradients_2/default_policy/Mean_1_grad/Shape*
_output_shapes
: *

index_type0
У
Cdefault_policy/gradients_2/default_policy/Mean_1_grad/DynamicStitchDynamicStitch;default_policy/gradients_2/default_policy/Mean_1_grad/range9default_policy/gradients_2/default_policy/Mean_1_grad/mod;default_policy/gradients_2/default_policy/Mean_1_grad/Shape:default_policy/gradients_2/default_policy/Mean_1_grad/ones*
N*
T0*N
_classD
B@loc:@default_policy/gradients_2/default_policy/Mean_1_grad/Shape*
_output_shapes
:

=default_policy/gradients_2/default_policy/Mean_1_grad/ReshapeReshapePdefault_policy/gradients_2/default_policy/mul_17_grad/tuple/control_dependency_1Cdefault_policy/gradients_2/default_policy/Mean_1_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:

Adefault_policy/gradients_2/default_policy/Mean_1_grad/BroadcastToBroadcastTo=default_policy/gradients_2/default_policy/Mean_1_grad/Reshape;default_policy/gradients_2/default_policy/Mean_1_grad/Shape*
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ

=default_policy/gradients_2/default_policy/Mean_1_grad/Shape_2Shape"default_policy/SquaredDifference_1*
T0*
_output_shapes
:*
out_type0

=default_policy/gradients_2/default_policy/Mean_1_grad/Shape_3Const*
_output_shapes
: *
dtype0*
valueB 

;default_policy/gradients_2/default_policy/Mean_1_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
ќ
:default_policy/gradients_2/default_policy/Mean_1_grad/ProdProd=default_policy/gradients_2/default_policy/Mean_1_grad/Shape_2;default_policy/gradients_2/default_policy/Mean_1_grad/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 

=default_policy/gradients_2/default_policy/Mean_1_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 

<default_policy/gradients_2/default_policy/Mean_1_grad/Prod_1Prod=default_policy/gradients_2/default_policy/Mean_1_grad/Shape_3=default_policy/gradients_2/default_policy/Mean_1_grad/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 

?default_policy/gradients_2/default_policy/Mean_1_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :
ш
=default_policy/gradients_2/default_policy/Mean_1_grad/MaximumMaximum<default_policy/gradients_2/default_policy/Mean_1_grad/Prod_1?default_policy/gradients_2/default_policy/Mean_1_grad/Maximum/y*
T0*
_output_shapes
: 
ц
>default_policy/gradients_2/default_policy/Mean_1_grad/floordivFloorDiv:default_policy/gradients_2/default_policy/Mean_1_grad/Prod=default_policy/gradients_2/default_policy/Mean_1_grad/Maximum*
T0*
_output_shapes
: 
Т
:default_policy/gradients_2/default_policy/Mean_1_grad/CastCast>default_policy/gradients_2/default_policy/Mean_1_grad/floordiv*

DstT0*

SrcT0*
Truncate( *
_output_shapes
: 

=default_policy/gradients_2/default_policy/Mean_1_grad/truedivRealDivAdefault_policy/gradients_2/default_policy/Mean_1_grad/BroadcastTo:default_policy/gradients_2/default_policy/Mean_1_grad/Cast*
T0*&
 _has_manual_control_dependencies(*#
_output_shapes
:џџџџџџџџџ
Ю
Idefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/scalarConst>^default_policy/gradients_2/default_policy/Mean_1_grad/truediv*
_output_shapes
: *
dtype0*
valueB
 *   @

Fdefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/MulMulIdefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/scalar=default_policy/gradients_2/default_policy/Mean_1_grad/truediv*
T0*#
_output_shapes
:џџџџџџџџџ
ђ
Fdefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/subSubdefault_policy/Squeeze_1default_policy/StopGradient>^default_policy/gradients_2/default_policy/Mean_1_grad/truediv*
T0*#
_output_shapes
:џџџџџџџџџ

Hdefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/mul_1MulFdefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/MulFdefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/sub*
T0*#
_output_shapes
:џџџџџџџџџ
 
Hdefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/ShapeShapedefault_policy/Squeeze_1*
T0*
_output_shapes
:*
out_type0
Ѕ
Jdefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/Shape_1Shapedefault_policy/StopGradient*
T0*
_output_shapes
:*
out_type0
Ф
Xdefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/BroadcastGradientArgsBroadcastGradientArgsHdefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/ShapeJdefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Б
Fdefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/SumSumHdefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/mul_1Xdefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Ы
Jdefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/ReshapeReshapeFdefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/SumHdefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*#
_output_shapes
:џџџџџџџџџ
Е
Hdefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/Sum_1SumHdefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/mul_1Zdefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Љ
Ldefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/Reshape_1ReshapeHdefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/Sum_1Jdefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ
ё
Fdefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/NegNegLdefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/Reshape_1*
T0*&
 _has_manual_control_dependencies(*#
_output_shapes
:џџџџџџџџџ

Sdefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/tuple/group_depsNoOpG^default_policy/gradients_2/default_policy/SquaredDifference_1_grad/NegK^default_policy/gradients_2/default_policy/SquaredDifference_1_grad/Reshape*&
 _has_manual_control_dependencies(

[default_policy/gradients_2/default_policy/SquaredDifference_1_grad/tuple/control_dependencyIdentityJdefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/ReshapeT^default_policy/gradients_2/default_policy/SquaredDifference_1_grad/tuple/group_deps*
T0*]
_classS
QOloc:@default_policy/gradients_2/default_policy/SquaredDifference_1_grad/Reshape*#
_output_shapes
:џџџџџџџџџ

]default_policy/gradients_2/default_policy/SquaredDifference_1_grad/tuple/control_dependency_1IdentityFdefault_policy/gradients_2/default_policy/SquaredDifference_1_grad/NegT^default_policy/gradients_2/default_policy/SquaredDifference_1_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@default_policy/gradients_2/default_policy/SquaredDifference_1_grad/Neg*#
_output_shapes
:џџџџџџџџџ
Ж
>default_policy/gradients_2/default_policy/Squeeze_1_grad/ShapeShape8default_policy/model_2_1/sequential_2/twin_q_out/BiasAdd*
T0*
_output_shapes
:*
out_type0
а
@default_policy/gradients_2/default_policy/Squeeze_1_grad/ReshapeReshape[default_policy/gradients_2/default_policy/SquaredDifference_1_grad/tuple/control_dependency>default_policy/gradients_2/default_policy/Squeeze_1_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ

ddefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_out/BiasAdd_grad/BiasAddGradBiasAddGrad@default_policy/gradients_2/default_policy/Squeeze_1_grad/Reshape*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:*
data_formatNHWC
У
idefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_out/BiasAdd_grad/tuple/group_depsNoOpA^default_policy/gradients_2/default_policy/Squeeze_1_grad/Reshapee^default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_out/BiasAdd_grad/BiasAddGrad*&
 _has_manual_control_dependencies(
В
qdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_out/BiasAdd_grad/tuple/control_dependencyIdentity@default_policy/gradients_2/default_policy/Squeeze_1_grad/Reshapej^default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_out/BiasAdd_grad/tuple/group_deps*
T0*S
_classI
GEloc:@default_policy/gradients_2/default_policy/Squeeze_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
я
sdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_out/BiasAdd_grad/tuple/control_dependency_1Identityddefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_out/BiasAdd_grad/BiasAddGradj^default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_out/BiasAdd_grad/tuple/group_deps*
T0*w
_classm
kiloc:@default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_out/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
Є
^default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_out/MatMul_grad/MatMulMatMulqdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_out/BiasAdd_grad/tuple/control_dependencyFdefault_policy/model_2_1/sequential_2/twin_q_out/MatMul/ReadVariableOp*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(

`default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_out/MatMul_grad/MatMul_1MatMul:default_policy/model_2_1/sequential_2/twin_q_hidden_1/Reluqdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_out/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:	*
transpose_a(*
transpose_b( 
м
hdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_out/MatMul_grad/tuple/group_depsNoOp_^default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_out/MatMul_grad/MatMula^default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_out/MatMul_grad/MatMul_1*&
 _has_manual_control_dependencies(
э
pdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_out/MatMul_grad/tuple/control_dependencyIdentity^default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_out/MatMul_grad/MatMuli^default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_out/MatMul_grad/tuple/group_deps*
T0*q
_classg
ecloc:@default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_out/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
ъ
rdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_out/MatMul_grad/tuple/control_dependency_1Identity`default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_out/MatMul_grad/MatMul_1i^default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_out/MatMul_grad/tuple/group_deps*
T0*s
_classi
geloc:@default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_out/MatMul_grad/MatMul_1*
_output_shapes
:	
ј
cdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/Relu_grad/ReluGradReluGradpdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_out/MatMul_grad/tuple/control_dependency:default_policy/model_2_1/sequential_2/twin_q_hidden_1/Relu*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:џџџџџџџџџ
Т
idefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/BiasAdd_grad/BiasAddGradBiasAddGradcdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/Relu_grad/ReluGrad*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:*
data_formatNHWC
№
ndefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/BiasAdd_grad/tuple/group_depsNoOpj^default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/BiasAdd_grad/BiasAddGradd^default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/Relu_grad/ReluGrad*&
 _has_manual_control_dependencies(

vdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/BiasAdd_grad/tuple/control_dependencyIdentitycdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/Relu_grad/ReluGrado^default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/BiasAdd_grad/tuple/group_deps*
T0*v
_classl
jhloc:@default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/Relu_grad/ReluGrad*(
_output_shapes
:џџџџџџџџџ

xdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/BiasAdd_grad/tuple/control_dependency_1Identityidefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/BiasAdd_grad/BiasAddGrado^default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/BiasAdd_grad/tuple/group_deps*
T0*|
_classr
pnloc:@default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Г
cdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/MatMul_grad/MatMulMatMulvdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/BiasAdd_grad/tuple/control_dependencyKdefault_policy/model_2_1/sequential_2/twin_q_hidden_1/MatMul/ReadVariableOp*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(

edefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/MatMul_grad/MatMul_1MatMul:default_policy/model_2_1/sequential_2/twin_q_hidden_0/Reluvdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
ы
mdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/MatMul_grad/tuple/group_depsNoOpd^default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/MatMul_grad/MatMulf^default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/MatMul_grad/MatMul_1*&
 _has_manual_control_dependencies(

udefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/MatMul_grad/tuple/control_dependencyIdentitycdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/MatMul_grad/MatMuln^default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/MatMul_grad/tuple/group_deps*
T0*v
_classl
jhloc:@default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
џ
wdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/MatMul_grad/tuple/control_dependency_1Identityedefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/MatMul_grad/MatMul_1n^default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/MatMul_grad/tuple/group_deps*
T0*x
_classn
ljloc:@default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/MatMul_grad/MatMul_1* 
_output_shapes
:

§
cdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/Relu_grad/ReluGradReluGradudefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/MatMul_grad/tuple/control_dependency:default_policy/model_2_1/sequential_2/twin_q_hidden_0/Relu*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:џџџџџџџџџ
Т
idefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/BiasAdd_grad/BiasAddGradBiasAddGradcdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/Relu_grad/ReluGrad*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:*
data_formatNHWC
№
ndefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/BiasAdd_grad/tuple/group_depsNoOpj^default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/BiasAdd_grad/BiasAddGradd^default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/Relu_grad/ReluGrad*&
 _has_manual_control_dependencies(

vdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/BiasAdd_grad/tuple/control_dependencyIdentitycdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/Relu_grad/ReluGrado^default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/BiasAdd_grad/tuple/group_deps*
T0*v
_classl
jhloc:@default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/Relu_grad/ReluGrad*(
_output_shapes
:џџџџџџџџџ

xdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/BiasAdd_grad/tuple/control_dependency_1Identityidefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/BiasAdd_grad/BiasAddGrado^default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/BiasAdd_grad/tuple/group_deps*
T0*|
_classr
pnloc:@default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Г
cdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/MatMul_grad/MatMulMatMulvdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/BiasAdd_grad/tuple/control_dependencyKdefault_policy/model_2_1/sequential_2/twin_q_hidden_0/MatMul/ReadVariableOp*
T0*&
 _has_manual_control_dependencies(*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(

edefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/MatMul_grad/MatMul_1MatMul:default_policy/model_2_1/sequential_2/concatenate_1/concatvdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/BiasAdd_grad/tuple/control_dependency*
T0*&
 _has_manual_control_dependencies(* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
ы
mdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/MatMul_grad/tuple/group_depsNoOpd^default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/MatMul_grad/MatMulf^default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/MatMul_grad/MatMul_1*&
 _has_manual_control_dependencies(

udefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/MatMul_grad/tuple/control_dependencyIdentitycdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/MatMul_grad/MatMuln^default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/MatMul_grad/tuple/group_deps*
T0*v
_classl
jhloc:@default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
џ
wdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/MatMul_grad/tuple/control_dependency_1Identityedefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/MatMul_grad/MatMul_1n^default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/MatMul_grad/tuple/group_deps*
T0*x
_classn
ljloc:@default_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/MatMul_grad/MatMul_1* 
_output_shapes
:

c
 default_policy/gradients_3/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
o
*default_policy/gradients_3/grad_ys_0/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
­
$default_policy/gradients_3/grad_ys_0Fill default_policy/gradients_3/Shape*default_policy/gradients_3/grad_ys_0/Const*
T0*
_output_shapes
: *

index_type0

6default_policy/gradients_3/default_policy/Neg_grad/NegNeg$default_policy/gradients_3/grad_ys_0*
T0*
_output_shapes
: 

Cdefault_policy/gradients_3/default_policy/Mean_2_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
ќ
=default_policy/gradients_3/default_policy/Mean_2_grad/ReshapeReshape6default_policy/gradients_3/default_policy/Neg_grad/NegCdefault_policy/gradients_3/default_policy/Mean_2_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:

;default_policy/gradients_3/default_policy/Mean_2_grad/ShapeShapedefault_policy/mul_18*
T0*
_output_shapes
:*
out_type0

:default_policy/gradients_3/default_policy/Mean_2_grad/TileTile=default_policy/gradients_3/default_policy/Mean_2_grad/Reshape;default_policy/gradients_3/default_policy/Mean_2_grad/Shape*
T0*

Tmultiples0*'
_output_shapes
:џџџџџџџџџ

=default_policy/gradients_3/default_policy/Mean_2_grad/Shape_1Shapedefault_policy/mul_18*
T0*
_output_shapes
:*
out_type0

=default_policy/gradients_3/default_policy/Mean_2_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 

;default_policy/gradients_3/default_policy/Mean_2_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
ќ
:default_policy/gradients_3/default_policy/Mean_2_grad/ProdProd=default_policy/gradients_3/default_policy/Mean_2_grad/Shape_1;default_policy/gradients_3/default_policy/Mean_2_grad/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 

=default_policy/gradients_3/default_policy/Mean_2_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 

<default_policy/gradients_3/default_policy/Mean_2_grad/Prod_1Prod=default_policy/gradients_3/default_policy/Mean_2_grad/Shape_2=default_policy/gradients_3/default_policy/Mean_2_grad/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 

?default_policy/gradients_3/default_policy/Mean_2_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :
ш
=default_policy/gradients_3/default_policy/Mean_2_grad/MaximumMaximum<default_policy/gradients_3/default_policy/Mean_2_grad/Prod_1?default_policy/gradients_3/default_policy/Mean_2_grad/Maximum/y*
T0*
_output_shapes
: 
ц
>default_policy/gradients_3/default_policy/Mean_2_grad/floordivFloorDiv:default_policy/gradients_3/default_policy/Mean_2_grad/Prod=default_policy/gradients_3/default_policy/Mean_2_grad/Maximum*
T0*
_output_shapes
: 
Т
:default_policy/gradients_3/default_policy/Mean_2_grad/CastCast>default_policy/gradients_3/default_policy/Mean_2_grad/floordiv*

DstT0*

SrcT0*
Truncate( *
_output_shapes
: 
ђ
=default_policy/gradients_3/default_policy/Mean_2_grad/truedivRealDiv:default_policy/gradients_3/default_policy/Mean_2_grad/Tile:default_policy/gradients_3/default_policy/Mean_2_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ

;default_policy/gradients_3/default_policy/mul_18_grad/ShapeShapedefault_policy/ReadVariableOp*
T0*
_output_shapes
: *
out_type0

=default_policy/gradients_3/default_policy/mul_18_grad/Shape_1Shapedefault_policy/StopGradient_1*
T0*
_output_shapes
:*
out_type0

Kdefault_policy/gradients_3/default_policy/mul_18_grad/BroadcastGradientArgsBroadcastGradientArgs;default_policy/gradients_3/default_policy/mul_18_grad/Shape=default_policy/gradients_3/default_policy/mul_18_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
а
9default_policy/gradients_3/default_policy/mul_18_grad/MulMul=default_policy/gradients_3/default_policy/Mean_2_grad/truedivdefault_policy/StopGradient_1*
T0*'
_output_shapes
:џџџџџџџџџ

9default_policy/gradients_3/default_policy/mul_18_grad/SumSum9default_policy/gradients_3/default_policy/mul_18_grad/MulKdefault_policy/gradients_3/default_policy/mul_18_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

=default_policy/gradients_3/default_policy/mul_18_grad/ReshapeReshape9default_policy/gradients_3/default_policy/mul_18_grad/Sum;default_policy/gradients_3/default_policy/mul_18_grad/Shape*
T0*
Tshape0*&
 _has_manual_control_dependencies(*
_output_shapes
: 
в
;default_policy/gradients_3/default_policy/mul_18_grad/Mul_1Muldefault_policy/ReadVariableOp=default_policy/gradients_3/default_policy/Mean_2_grad/truediv*
T0*'
_output_shapes
:џџџџџџџџџ

;default_policy/gradients_3/default_policy/mul_18_grad/Sum_1Sum;default_policy/gradients_3/default_policy/mul_18_grad/Mul_1Mdefault_policy/gradients_3/default_policy/mul_18_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
Ў
?default_policy/gradients_3/default_policy/mul_18_grad/Reshape_1Reshape;default_policy/gradients_3/default_policy/mul_18_grad/Sum_1=default_policy/gradients_3/default_policy/mul_18_grad/Shape_1*
T0*
Tshape0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџ
ј
Fdefault_policy/gradients_3/default_policy/mul_18_grad/tuple/group_depsNoOp>^default_policy/gradients_3/default_policy/mul_18_grad/Reshape@^default_policy/gradients_3/default_policy/mul_18_grad/Reshape_1*&
 _has_manual_control_dependencies(
е
Ndefault_policy/gradients_3/default_policy/mul_18_grad/tuple/control_dependencyIdentity=default_policy/gradients_3/default_policy/mul_18_grad/ReshapeG^default_policy/gradients_3/default_policy/mul_18_grad/tuple/group_deps*
T0*P
_classF
DBloc:@default_policy/gradients_3/default_policy/mul_18_grad/Reshape*
_output_shapes
: 
ь
Pdefault_policy/gradients_3/default_policy/mul_18_grad/tuple/control_dependency_1Identity?default_policy/gradients_3/default_policy/mul_18_grad/Reshape_1G^default_policy/gradients_3/default_policy/mul_18_grad/tuple/group_deps*
T0*R
_classH
FDloc:@default_policy/gradients_3/default_policy/mul_18_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Д
default_policy/IdentityIdentityddefault_policy/gradients/default_policy/sequential_7/action_1/MatMul_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:

В
default_policy/Identity_1Identityedefault_policy/gradients/default_policy/sequential_7/action_1/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:
Ж
default_policy/Identity_2Identityddefault_policy/gradients/default_policy/sequential_7/action_2/MatMul_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:

В
default_policy/Identity_3Identityedefault_policy/gradients/default_policy/sequential_7/action_2/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:
З
default_policy/Identity_4Identityfdefault_policy/gradients/default_policy/sequential_7/action_out/MatMul_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	
Г
default_policy/Identity_5Identitygdefault_policy/gradients/default_policy/sequential_7/action_out/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
:
Ф
default_policy/Identity_6Identityrdefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/MatMul_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:

Р
default_policy/Identity_7Identitysdefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_0/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:
Ф
default_policy/Identity_8Identityrdefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/MatMul_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:

Р
default_policy/Identity_9Identitysdefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_hidden_1/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:
П
default_policy/Identity_10Identitymdefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_out/MatMul_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	
Л
default_policy/Identity_11Identityndefault_policy/gradients_1/default_policy/model_1_1/sequential_1/q_out/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
:
Ъ
default_policy/Identity_12Identitywdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/MatMul_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:

Ц
default_policy/Identity_13Identityxdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_0/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:
Ъ
default_policy/Identity_14Identitywdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/MatMul_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:

Ц
default_policy/Identity_15Identityxdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_hidden_1/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:
Ф
default_policy/Identity_16Identityrdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_out/MatMul_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	
Р
default_policy/Identity_17Identitysdefault_policy/gradients_2/default_policy/model_2_1/sequential_2/twin_q_out/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
:

default_policy/Identity_18IdentityNdefault_policy/gradients_3/default_policy/mul_18_grad/tuple/control_dependency*
T0*
_output_shapes
: 

default_policy/ReadVariableOp_1ReadVariableOpdefault_policy/value_out/kernel*
_output_shapes
:	*
dtype0

:default_policy/Placeholder_default_policy/value_out/kernelPlaceholder*
_output_shapes
:	*
dtype0*
shape:	
л
default_policy/AssignVariableOpAssignVariableOpdefault_policy/value_out/kernel:default_policy/Placeholder_default_policy/value_out/kernel*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Ђ
default_policy/ReadVariableOp_2ReadVariableOpdefault_policy/value_out/kernel ^default_policy/AssignVariableOp*
_output_shapes
:	*
dtype0
y
default_policy/ReadVariableOp_3ReadVariableOpdefault_policy/value_out/bias*
_output_shapes
:*
dtype0

8default_policy/Placeholder_default_policy/value_out/biasPlaceholder*
_output_shapes
:*
dtype0*
shape:
й
!default_policy/AssignVariableOp_1AssignVariableOpdefault_policy/value_out/bias8default_policy/Placeholder_default_policy/value_out/bias*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(

default_policy/ReadVariableOp_4ReadVariableOpdefault_policy/value_out/bias"^default_policy/AssignVariableOp_1*
_output_shapes
:*
dtype0

default_policy/ReadVariableOp_5ReadVariableOp)default_policy/sequential/action_1/kernel* 
_output_shapes
:
*
dtype0

Ddefault_policy/Placeholder_default_policy/sequential/action_1/kernelPlaceholder* 
_output_shapes
:
*
dtype0*
shape:

ё
!default_policy/AssignVariableOp_2AssignVariableOp)default_policy/sequential/action_1/kernelDdefault_policy/Placeholder_default_policy/sequential/action_1/kernel*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Џ
default_policy/ReadVariableOp_6ReadVariableOp)default_policy/sequential/action_1/kernel"^default_policy/AssignVariableOp_2* 
_output_shapes
:
*
dtype0

default_policy/ReadVariableOp_7ReadVariableOp'default_policy/sequential/action_1/bias*
_output_shapes	
:*
dtype0

Bdefault_policy/Placeholder_default_policy/sequential/action_1/biasPlaceholder*
_output_shapes	
:*
dtype0*
shape:
э
!default_policy/AssignVariableOp_3AssignVariableOp'default_policy/sequential/action_1/biasBdefault_policy/Placeholder_default_policy/sequential/action_1/bias*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Ј
default_policy/ReadVariableOp_8ReadVariableOp'default_policy/sequential/action_1/bias"^default_policy/AssignVariableOp_3*
_output_shapes	
:*
dtype0

default_policy/ReadVariableOp_9ReadVariableOp)default_policy/sequential/action_2/kernel* 
_output_shapes
:
*
dtype0

Ddefault_policy/Placeholder_default_policy/sequential/action_2/kernelPlaceholder* 
_output_shapes
:
*
dtype0*
shape:

ё
!default_policy/AssignVariableOp_4AssignVariableOp)default_policy/sequential/action_2/kernelDdefault_policy/Placeholder_default_policy/sequential/action_2/kernel*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
А
 default_policy/ReadVariableOp_10ReadVariableOp)default_policy/sequential/action_2/kernel"^default_policy/AssignVariableOp_4* 
_output_shapes
:
*
dtype0

 default_policy/ReadVariableOp_11ReadVariableOp'default_policy/sequential/action_2/bias*
_output_shapes	
:*
dtype0

Bdefault_policy/Placeholder_default_policy/sequential/action_2/biasPlaceholder*
_output_shapes	
:*
dtype0*
shape:
э
!default_policy/AssignVariableOp_5AssignVariableOp'default_policy/sequential/action_2/biasBdefault_policy/Placeholder_default_policy/sequential/action_2/bias*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Љ
 default_policy/ReadVariableOp_12ReadVariableOp'default_policy/sequential/action_2/bias"^default_policy/AssignVariableOp_5*
_output_shapes	
:*
dtype0

 default_policy/ReadVariableOp_13ReadVariableOp+default_policy/sequential/action_out/kernel*
_output_shapes
:	*
dtype0

Fdefault_policy/Placeholder_default_policy/sequential/action_out/kernelPlaceholder*
_output_shapes
:	*
dtype0*
shape:	
ѕ
!default_policy/AssignVariableOp_6AssignVariableOp+default_policy/sequential/action_out/kernelFdefault_policy/Placeholder_default_policy/sequential/action_out/kernel*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Б
 default_policy/ReadVariableOp_14ReadVariableOp+default_policy/sequential/action_out/kernel"^default_policy/AssignVariableOp_6*
_output_shapes
:	*
dtype0

 default_policy/ReadVariableOp_15ReadVariableOp)default_policy/sequential/action_out/bias*
_output_shapes
:*
dtype0

Ddefault_policy/Placeholder_default_policy/sequential/action_out/biasPlaceholder*
_output_shapes
:*
dtype0*
shape:
ё
!default_policy/AssignVariableOp_7AssignVariableOp)default_policy/sequential/action_out/biasDdefault_policy/Placeholder_default_policy/sequential/action_out/bias*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Њ
 default_policy/ReadVariableOp_16ReadVariableOp)default_policy/sequential/action_out/bias"^default_policy/AssignVariableOp_7*
_output_shapes
:*
dtype0

 default_policy/ReadVariableOp_17ReadVariableOp-default_policy/sequential_1/q_hidden_0/kernel* 
_output_shapes
:
*
dtype0

Hdefault_policy/Placeholder_default_policy/sequential_1/q_hidden_0/kernelPlaceholder* 
_output_shapes
:
*
dtype0*
shape:

љ
!default_policy/AssignVariableOp_8AssignVariableOp-default_policy/sequential_1/q_hidden_0/kernelHdefault_policy/Placeholder_default_policy/sequential_1/q_hidden_0/kernel*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Д
 default_policy/ReadVariableOp_18ReadVariableOp-default_policy/sequential_1/q_hidden_0/kernel"^default_policy/AssignVariableOp_8* 
_output_shapes
:
*
dtype0

 default_policy/ReadVariableOp_19ReadVariableOp+default_policy/sequential_1/q_hidden_0/bias*
_output_shapes	
:*
dtype0

Fdefault_policy/Placeholder_default_policy/sequential_1/q_hidden_0/biasPlaceholder*
_output_shapes	
:*
dtype0*
shape:
ѕ
!default_policy/AssignVariableOp_9AssignVariableOp+default_policy/sequential_1/q_hidden_0/biasFdefault_policy/Placeholder_default_policy/sequential_1/q_hidden_0/bias*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
­
 default_policy/ReadVariableOp_20ReadVariableOp+default_policy/sequential_1/q_hidden_0/bias"^default_policy/AssignVariableOp_9*
_output_shapes	
:*
dtype0

 default_policy/ReadVariableOp_21ReadVariableOp-default_policy/sequential_1/q_hidden_1/kernel* 
_output_shapes
:
*
dtype0

Hdefault_policy/Placeholder_default_policy/sequential_1/q_hidden_1/kernelPlaceholder* 
_output_shapes
:
*
dtype0*
shape:

њ
"default_policy/AssignVariableOp_10AssignVariableOp-default_policy/sequential_1/q_hidden_1/kernelHdefault_policy/Placeholder_default_policy/sequential_1/q_hidden_1/kernel*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Е
 default_policy/ReadVariableOp_22ReadVariableOp-default_policy/sequential_1/q_hidden_1/kernel#^default_policy/AssignVariableOp_10* 
_output_shapes
:
*
dtype0

 default_policy/ReadVariableOp_23ReadVariableOp+default_policy/sequential_1/q_hidden_1/bias*
_output_shapes	
:*
dtype0

Fdefault_policy/Placeholder_default_policy/sequential_1/q_hidden_1/biasPlaceholder*
_output_shapes	
:*
dtype0*
shape:
і
"default_policy/AssignVariableOp_11AssignVariableOp+default_policy/sequential_1/q_hidden_1/biasFdefault_policy/Placeholder_default_policy/sequential_1/q_hidden_1/bias*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Ў
 default_policy/ReadVariableOp_24ReadVariableOp+default_policy/sequential_1/q_hidden_1/bias#^default_policy/AssignVariableOp_11*
_output_shapes	
:*
dtype0

 default_policy/ReadVariableOp_25ReadVariableOp(default_policy/sequential_1/q_out/kernel*
_output_shapes
:	*
dtype0

Cdefault_policy/Placeholder_default_policy/sequential_1/q_out/kernelPlaceholder*
_output_shapes
:	*
dtype0*
shape:	
№
"default_policy/AssignVariableOp_12AssignVariableOp(default_policy/sequential_1/q_out/kernelCdefault_policy/Placeholder_default_policy/sequential_1/q_out/kernel*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Џ
 default_policy/ReadVariableOp_26ReadVariableOp(default_policy/sequential_1/q_out/kernel#^default_policy/AssignVariableOp_12*
_output_shapes
:	*
dtype0

 default_policy/ReadVariableOp_27ReadVariableOp&default_policy/sequential_1/q_out/bias*
_output_shapes
:*
dtype0

Adefault_policy/Placeholder_default_policy/sequential_1/q_out/biasPlaceholder*
_output_shapes
:*
dtype0*
shape:
ь
"default_policy/AssignVariableOp_13AssignVariableOp&default_policy/sequential_1/q_out/biasAdefault_policy/Placeholder_default_policy/sequential_1/q_out/bias*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Ј
 default_policy/ReadVariableOp_28ReadVariableOp&default_policy/sequential_1/q_out/bias#^default_policy/AssignVariableOp_13*
_output_shapes
:*
dtype0

 default_policy/ReadVariableOp_29ReadVariableOp2default_policy/sequential_2/twin_q_hidden_0/kernel* 
_output_shapes
:
*
dtype0
Ђ
Mdefault_policy/Placeholder_default_policy/sequential_2/twin_q_hidden_0/kernelPlaceholder* 
_output_shapes
:
*
dtype0*
shape:


"default_policy/AssignVariableOp_14AssignVariableOp2default_policy/sequential_2/twin_q_hidden_0/kernelMdefault_policy/Placeholder_default_policy/sequential_2/twin_q_hidden_0/kernel*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
К
 default_policy/ReadVariableOp_30ReadVariableOp2default_policy/sequential_2/twin_q_hidden_0/kernel#^default_policy/AssignVariableOp_14* 
_output_shapes
:
*
dtype0

 default_policy/ReadVariableOp_31ReadVariableOp0default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes	
:*
dtype0

Kdefault_policy/Placeholder_default_policy/sequential_2/twin_q_hidden_0/biasPlaceholder*
_output_shapes	
:*
dtype0*
shape:

"default_policy/AssignVariableOp_15AssignVariableOp0default_policy/sequential_2/twin_q_hidden_0/biasKdefault_policy/Placeholder_default_policy/sequential_2/twin_q_hidden_0/bias*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Г
 default_policy/ReadVariableOp_32ReadVariableOp0default_policy/sequential_2/twin_q_hidden_0/bias#^default_policy/AssignVariableOp_15*
_output_shapes	
:*
dtype0

 default_policy/ReadVariableOp_33ReadVariableOp2default_policy/sequential_2/twin_q_hidden_1/kernel* 
_output_shapes
:
*
dtype0
Ђ
Mdefault_policy/Placeholder_default_policy/sequential_2/twin_q_hidden_1/kernelPlaceholder* 
_output_shapes
:
*
dtype0*
shape:


"default_policy/AssignVariableOp_16AssignVariableOp2default_policy/sequential_2/twin_q_hidden_1/kernelMdefault_policy/Placeholder_default_policy/sequential_2/twin_q_hidden_1/kernel*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
К
 default_policy/ReadVariableOp_34ReadVariableOp2default_policy/sequential_2/twin_q_hidden_1/kernel#^default_policy/AssignVariableOp_16* 
_output_shapes
:
*
dtype0

 default_policy/ReadVariableOp_35ReadVariableOp0default_policy/sequential_2/twin_q_hidden_1/bias*
_output_shapes	
:*
dtype0

Kdefault_policy/Placeholder_default_policy/sequential_2/twin_q_hidden_1/biasPlaceholder*
_output_shapes	
:*
dtype0*
shape:

"default_policy/AssignVariableOp_17AssignVariableOp0default_policy/sequential_2/twin_q_hidden_1/biasKdefault_policy/Placeholder_default_policy/sequential_2/twin_q_hidden_1/bias*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Г
 default_policy/ReadVariableOp_36ReadVariableOp0default_policy/sequential_2/twin_q_hidden_1/bias#^default_policy/AssignVariableOp_17*
_output_shapes	
:*
dtype0

 default_policy/ReadVariableOp_37ReadVariableOp-default_policy/sequential_2/twin_q_out/kernel*
_output_shapes
:	*
dtype0

Hdefault_policy/Placeholder_default_policy/sequential_2/twin_q_out/kernelPlaceholder*
_output_shapes
:	*
dtype0*
shape:	
њ
"default_policy/AssignVariableOp_18AssignVariableOp-default_policy/sequential_2/twin_q_out/kernelHdefault_policy/Placeholder_default_policy/sequential_2/twin_q_out/kernel*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Д
 default_policy/ReadVariableOp_38ReadVariableOp-default_policy/sequential_2/twin_q_out/kernel#^default_policy/AssignVariableOp_18*
_output_shapes
:	*
dtype0

 default_policy/ReadVariableOp_39ReadVariableOp+default_policy/sequential_2/twin_q_out/bias*
_output_shapes
:*
dtype0

Fdefault_policy/Placeholder_default_policy/sequential_2/twin_q_out/biasPlaceholder*
_output_shapes
:*
dtype0*
shape:
і
"default_policy/AssignVariableOp_19AssignVariableOp+default_policy/sequential_2/twin_q_out/biasFdefault_policy/Placeholder_default_policy/sequential_2/twin_q_out/bias*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
­
 default_policy/ReadVariableOp_40ReadVariableOp+default_policy/sequential_2/twin_q_out/bias#^default_policy/AssignVariableOp_19*
_output_shapes
:*
dtype0
q
 default_policy/ReadVariableOp_41ReadVariableOpdefault_policy/log_alpha*
_output_shapes
: *
dtype0
t
3default_policy/Placeholder_default_policy/log_alphaPlaceholder*
_output_shapes
: *
dtype0*
shape: 
а
"default_policy/AssignVariableOp_20AssignVariableOpdefault_policy/log_alpha3default_policy/Placeholder_default_policy/log_alpha*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(

 default_policy/ReadVariableOp_42ReadVariableOpdefault_policy/log_alpha#^default_policy/AssignVariableOp_20*
_output_shapes
: *
dtype0

 default_policy/ReadVariableOp_43ReadVariableOp!default_policy/value_out_1/kernel*
_output_shapes
:	*
dtype0

<default_policy/Placeholder_default_policy/value_out_1/kernelPlaceholder*
_output_shapes
:	*
dtype0*
shape:	
т
"default_policy/AssignVariableOp_21AssignVariableOp!default_policy/value_out_1/kernel<default_policy/Placeholder_default_policy/value_out_1/kernel*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Ј
 default_policy/ReadVariableOp_44ReadVariableOp!default_policy/value_out_1/kernel#^default_policy/AssignVariableOp_21*
_output_shapes
:	*
dtype0
|
 default_policy/ReadVariableOp_45ReadVariableOpdefault_policy/value_out_1/bias*
_output_shapes
:*
dtype0

:default_policy/Placeholder_default_policy/value_out_1/biasPlaceholder*
_output_shapes
:*
dtype0*
shape:
о
"default_policy/AssignVariableOp_22AssignVariableOpdefault_policy/value_out_1/bias:default_policy/Placeholder_default_policy/value_out_1/bias*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Ё
 default_policy/ReadVariableOp_46ReadVariableOpdefault_policy/value_out_1/bias#^default_policy/AssignVariableOp_22*
_output_shapes
:*
dtype0

 default_policy/ReadVariableOp_47ReadVariableOp+default_policy/sequential_3/action_1/kernel* 
_output_shapes
:
*
dtype0

Fdefault_policy/Placeholder_default_policy/sequential_3/action_1/kernelPlaceholder* 
_output_shapes
:
*
dtype0*
shape:

і
"default_policy/AssignVariableOp_23AssignVariableOp+default_policy/sequential_3/action_1/kernelFdefault_policy/Placeholder_default_policy/sequential_3/action_1/kernel*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Г
 default_policy/ReadVariableOp_48ReadVariableOp+default_policy/sequential_3/action_1/kernel#^default_policy/AssignVariableOp_23* 
_output_shapes
:
*
dtype0

 default_policy/ReadVariableOp_49ReadVariableOp)default_policy/sequential_3/action_1/bias*
_output_shapes	
:*
dtype0

Ddefault_policy/Placeholder_default_policy/sequential_3/action_1/biasPlaceholder*
_output_shapes	
:*
dtype0*
shape:
ђ
"default_policy/AssignVariableOp_24AssignVariableOp)default_policy/sequential_3/action_1/biasDdefault_policy/Placeholder_default_policy/sequential_3/action_1/bias*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Ќ
 default_policy/ReadVariableOp_50ReadVariableOp)default_policy/sequential_3/action_1/bias#^default_policy/AssignVariableOp_24*
_output_shapes	
:*
dtype0

 default_policy/ReadVariableOp_51ReadVariableOp+default_policy/sequential_3/action_2/kernel* 
_output_shapes
:
*
dtype0

Fdefault_policy/Placeholder_default_policy/sequential_3/action_2/kernelPlaceholder* 
_output_shapes
:
*
dtype0*
shape:

і
"default_policy/AssignVariableOp_25AssignVariableOp+default_policy/sequential_3/action_2/kernelFdefault_policy/Placeholder_default_policy/sequential_3/action_2/kernel*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Г
 default_policy/ReadVariableOp_52ReadVariableOp+default_policy/sequential_3/action_2/kernel#^default_policy/AssignVariableOp_25* 
_output_shapes
:
*
dtype0

 default_policy/ReadVariableOp_53ReadVariableOp)default_policy/sequential_3/action_2/bias*
_output_shapes	
:*
dtype0

Ddefault_policy/Placeholder_default_policy/sequential_3/action_2/biasPlaceholder*
_output_shapes	
:*
dtype0*
shape:
ђ
"default_policy/AssignVariableOp_26AssignVariableOp)default_policy/sequential_3/action_2/biasDdefault_policy/Placeholder_default_policy/sequential_3/action_2/bias*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Ќ
 default_policy/ReadVariableOp_54ReadVariableOp)default_policy/sequential_3/action_2/bias#^default_policy/AssignVariableOp_26*
_output_shapes	
:*
dtype0

 default_policy/ReadVariableOp_55ReadVariableOp-default_policy/sequential_3/action_out/kernel*
_output_shapes
:	*
dtype0

Hdefault_policy/Placeholder_default_policy/sequential_3/action_out/kernelPlaceholder*
_output_shapes
:	*
dtype0*
shape:	
њ
"default_policy/AssignVariableOp_27AssignVariableOp-default_policy/sequential_3/action_out/kernelHdefault_policy/Placeholder_default_policy/sequential_3/action_out/kernel*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Д
 default_policy/ReadVariableOp_56ReadVariableOp-default_policy/sequential_3/action_out/kernel#^default_policy/AssignVariableOp_27*
_output_shapes
:	*
dtype0

 default_policy/ReadVariableOp_57ReadVariableOp+default_policy/sequential_3/action_out/bias*
_output_shapes
:*
dtype0

Fdefault_policy/Placeholder_default_policy/sequential_3/action_out/biasPlaceholder*
_output_shapes
:*
dtype0*
shape:
і
"default_policy/AssignVariableOp_28AssignVariableOp+default_policy/sequential_3/action_out/biasFdefault_policy/Placeholder_default_policy/sequential_3/action_out/bias*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
­
 default_policy/ReadVariableOp_58ReadVariableOp+default_policy/sequential_3/action_out/bias#^default_policy/AssignVariableOp_28*
_output_shapes
:*
dtype0

 default_policy/ReadVariableOp_59ReadVariableOp-default_policy/sequential_4/q_hidden_0/kernel* 
_output_shapes
:
*
dtype0

Hdefault_policy/Placeholder_default_policy/sequential_4/q_hidden_0/kernelPlaceholder* 
_output_shapes
:
*
dtype0*
shape:

њ
"default_policy/AssignVariableOp_29AssignVariableOp-default_policy/sequential_4/q_hidden_0/kernelHdefault_policy/Placeholder_default_policy/sequential_4/q_hidden_0/kernel*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Е
 default_policy/ReadVariableOp_60ReadVariableOp-default_policy/sequential_4/q_hidden_0/kernel#^default_policy/AssignVariableOp_29* 
_output_shapes
:
*
dtype0

 default_policy/ReadVariableOp_61ReadVariableOp+default_policy/sequential_4/q_hidden_0/bias*
_output_shapes	
:*
dtype0

Fdefault_policy/Placeholder_default_policy/sequential_4/q_hidden_0/biasPlaceholder*
_output_shapes	
:*
dtype0*
shape:
і
"default_policy/AssignVariableOp_30AssignVariableOp+default_policy/sequential_4/q_hidden_0/biasFdefault_policy/Placeholder_default_policy/sequential_4/q_hidden_0/bias*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Ў
 default_policy/ReadVariableOp_62ReadVariableOp+default_policy/sequential_4/q_hidden_0/bias#^default_policy/AssignVariableOp_30*
_output_shapes	
:*
dtype0

 default_policy/ReadVariableOp_63ReadVariableOp-default_policy/sequential_4/q_hidden_1/kernel* 
_output_shapes
:
*
dtype0

Hdefault_policy/Placeholder_default_policy/sequential_4/q_hidden_1/kernelPlaceholder* 
_output_shapes
:
*
dtype0*
shape:

њ
"default_policy/AssignVariableOp_31AssignVariableOp-default_policy/sequential_4/q_hidden_1/kernelHdefault_policy/Placeholder_default_policy/sequential_4/q_hidden_1/kernel*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Е
 default_policy/ReadVariableOp_64ReadVariableOp-default_policy/sequential_4/q_hidden_1/kernel#^default_policy/AssignVariableOp_31* 
_output_shapes
:
*
dtype0

 default_policy/ReadVariableOp_65ReadVariableOp+default_policy/sequential_4/q_hidden_1/bias*
_output_shapes	
:*
dtype0

Fdefault_policy/Placeholder_default_policy/sequential_4/q_hidden_1/biasPlaceholder*
_output_shapes	
:*
dtype0*
shape:
і
"default_policy/AssignVariableOp_32AssignVariableOp+default_policy/sequential_4/q_hidden_1/biasFdefault_policy/Placeholder_default_policy/sequential_4/q_hidden_1/bias*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Ў
 default_policy/ReadVariableOp_66ReadVariableOp+default_policy/sequential_4/q_hidden_1/bias#^default_policy/AssignVariableOp_32*
_output_shapes	
:*
dtype0

 default_policy/ReadVariableOp_67ReadVariableOp(default_policy/sequential_4/q_out/kernel*
_output_shapes
:	*
dtype0

Cdefault_policy/Placeholder_default_policy/sequential_4/q_out/kernelPlaceholder*
_output_shapes
:	*
dtype0*
shape:	
№
"default_policy/AssignVariableOp_33AssignVariableOp(default_policy/sequential_4/q_out/kernelCdefault_policy/Placeholder_default_policy/sequential_4/q_out/kernel*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Џ
 default_policy/ReadVariableOp_68ReadVariableOp(default_policy/sequential_4/q_out/kernel#^default_policy/AssignVariableOp_33*
_output_shapes
:	*
dtype0

 default_policy/ReadVariableOp_69ReadVariableOp&default_policy/sequential_4/q_out/bias*
_output_shapes
:*
dtype0

Adefault_policy/Placeholder_default_policy/sequential_4/q_out/biasPlaceholder*
_output_shapes
:*
dtype0*
shape:
ь
"default_policy/AssignVariableOp_34AssignVariableOp&default_policy/sequential_4/q_out/biasAdefault_policy/Placeholder_default_policy/sequential_4/q_out/bias*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Ј
 default_policy/ReadVariableOp_70ReadVariableOp&default_policy/sequential_4/q_out/bias#^default_policy/AssignVariableOp_34*
_output_shapes
:*
dtype0

 default_policy/ReadVariableOp_71ReadVariableOp2default_policy/sequential_5/twin_q_hidden_0/kernel* 
_output_shapes
:
*
dtype0
Ђ
Mdefault_policy/Placeholder_default_policy/sequential_5/twin_q_hidden_0/kernelPlaceholder* 
_output_shapes
:
*
dtype0*
shape:


"default_policy/AssignVariableOp_35AssignVariableOp2default_policy/sequential_5/twin_q_hidden_0/kernelMdefault_policy/Placeholder_default_policy/sequential_5/twin_q_hidden_0/kernel*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
К
 default_policy/ReadVariableOp_72ReadVariableOp2default_policy/sequential_5/twin_q_hidden_0/kernel#^default_policy/AssignVariableOp_35* 
_output_shapes
:
*
dtype0

 default_policy/ReadVariableOp_73ReadVariableOp0default_policy/sequential_5/twin_q_hidden_0/bias*
_output_shapes	
:*
dtype0

Kdefault_policy/Placeholder_default_policy/sequential_5/twin_q_hidden_0/biasPlaceholder*
_output_shapes	
:*
dtype0*
shape:

"default_policy/AssignVariableOp_36AssignVariableOp0default_policy/sequential_5/twin_q_hidden_0/biasKdefault_policy/Placeholder_default_policy/sequential_5/twin_q_hidden_0/bias*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Г
 default_policy/ReadVariableOp_74ReadVariableOp0default_policy/sequential_5/twin_q_hidden_0/bias#^default_policy/AssignVariableOp_36*
_output_shapes	
:*
dtype0

 default_policy/ReadVariableOp_75ReadVariableOp2default_policy/sequential_5/twin_q_hidden_1/kernel* 
_output_shapes
:
*
dtype0
Ђ
Mdefault_policy/Placeholder_default_policy/sequential_5/twin_q_hidden_1/kernelPlaceholder* 
_output_shapes
:
*
dtype0*
shape:


"default_policy/AssignVariableOp_37AssignVariableOp2default_policy/sequential_5/twin_q_hidden_1/kernelMdefault_policy/Placeholder_default_policy/sequential_5/twin_q_hidden_1/kernel*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
К
 default_policy/ReadVariableOp_76ReadVariableOp2default_policy/sequential_5/twin_q_hidden_1/kernel#^default_policy/AssignVariableOp_37* 
_output_shapes
:
*
dtype0

 default_policy/ReadVariableOp_77ReadVariableOp0default_policy/sequential_5/twin_q_hidden_1/bias*
_output_shapes	
:*
dtype0

Kdefault_policy/Placeholder_default_policy/sequential_5/twin_q_hidden_1/biasPlaceholder*
_output_shapes	
:*
dtype0*
shape:

"default_policy/AssignVariableOp_38AssignVariableOp0default_policy/sequential_5/twin_q_hidden_1/biasKdefault_policy/Placeholder_default_policy/sequential_5/twin_q_hidden_1/bias*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Г
 default_policy/ReadVariableOp_78ReadVariableOp0default_policy/sequential_5/twin_q_hidden_1/bias#^default_policy/AssignVariableOp_38*
_output_shapes	
:*
dtype0

 default_policy/ReadVariableOp_79ReadVariableOp-default_policy/sequential_5/twin_q_out/kernel*
_output_shapes
:	*
dtype0

Hdefault_policy/Placeholder_default_policy/sequential_5/twin_q_out/kernelPlaceholder*
_output_shapes
:	*
dtype0*
shape:	
њ
"default_policy/AssignVariableOp_39AssignVariableOp-default_policy/sequential_5/twin_q_out/kernelHdefault_policy/Placeholder_default_policy/sequential_5/twin_q_out/kernel*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Д
 default_policy/ReadVariableOp_80ReadVariableOp-default_policy/sequential_5/twin_q_out/kernel#^default_policy/AssignVariableOp_39*
_output_shapes
:	*
dtype0

 default_policy/ReadVariableOp_81ReadVariableOp+default_policy/sequential_5/twin_q_out/bias*
_output_shapes
:*
dtype0

Fdefault_policy/Placeholder_default_policy/sequential_5/twin_q_out/biasPlaceholder*
_output_shapes
:*
dtype0*
shape:
і
"default_policy/AssignVariableOp_40AssignVariableOp+default_policy/sequential_5/twin_q_out/biasFdefault_policy/Placeholder_default_policy/sequential_5/twin_q_out/bias*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
­
 default_policy/ReadVariableOp_82ReadVariableOp+default_policy/sequential_5/twin_q_out/bias#^default_policy/AssignVariableOp_40*
_output_shapes
:*
dtype0
s
 default_policy/ReadVariableOp_83ReadVariableOpdefault_policy/log_alpha_1*
_output_shapes
: *
dtype0
v
5default_policy/Placeholder_default_policy/log_alpha_1Placeholder*
_output_shapes
: *
dtype0*
shape: 
д
"default_policy/AssignVariableOp_41AssignVariableOpdefault_policy/log_alpha_15default_policy/Placeholder_default_policy/log_alpha_1*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(

 default_policy/ReadVariableOp_84ReadVariableOpdefault_policy/log_alpha_1#^default_policy/AssignVariableOp_41*
_output_shapes
: *
dtype0
Е
4default_policy/beta1_power/Initializer/initial_valueConst*:
_class0
.,loc:@default_policy/sequential/action_1/bias*
_output_shapes
: *
dtype0*
valueB
 *fff?
ь
default_policy/beta1_powerVarHandleOp*:
_class0
.,loc:@default_policy/sequential/action_1/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *+
shared_namedefault_policy/beta1_power
С
;default_policy/beta1_power/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/beta1_power*:
_class0
.,loc:@default_policy/sequential/action_1/bias*
_output_shapes
: 
в
!default_policy/beta1_power/AssignAssignVariableOpdefault_policy/beta1_power4default_policy/beta1_power/Initializer/initial_value*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Н
.default_policy/beta1_power/Read/ReadVariableOpReadVariableOpdefault_policy/beta1_power*:
_class0
.,loc:@default_policy/sequential/action_1/bias*
_output_shapes
: *
dtype0
Е
4default_policy/beta2_power/Initializer/initial_valueConst*:
_class0
.,loc:@default_policy/sequential/action_1/bias*
_output_shapes
: *
dtype0*
valueB
 *wО?
ь
default_policy/beta2_powerVarHandleOp*:
_class0
.,loc:@default_policy/sequential/action_1/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *+
shared_namedefault_policy/beta2_power
С
;default_policy/beta2_power/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/beta2_power*:
_class0
.,loc:@default_policy/sequential/action_1/bias*
_output_shapes
: 
в
!default_policy/beta2_power/AssignAssignVariableOpdefault_policy/beta2_power4default_policy/beta2_power/Initializer/initial_value*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Н
.default_policy/beta2_power/Read/ReadVariableOpReadVariableOpdefault_policy/beta2_power*:
_class0
.,loc:@default_policy/sequential/action_1/bias*
_output_shapes
: *
dtype0
ю
_default_policy/default_policy/sequential/action_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*<
_class2
0.loc:@default_policy/sequential/action_1/kernel*
_output_shapes
:*
dtype0*
valueB"     
и
Udefault_policy/default_policy/sequential/action_1/kernel/Adam/Initializer/zeros/ConstConst*<
_class2
0.loc:@default_policy/sequential/action_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *    

Odefault_policy/default_policy/sequential/action_1/kernel/Adam/Initializer/zerosFill_default_policy/default_policy/sequential/action_1/kernel/Adam/Initializer/zeros/shape_as_tensorUdefault_policy/default_policy/sequential/action_1/kernel/Adam/Initializer/zeros/Const*
T0*<
_class2
0.loc:@default_policy/sequential/action_1/kernel* 
_output_shapes
:
*

index_type0
О
=default_policy/default_policy/sequential/action_1/kernel/AdamVarHandleOp*<
_class2
0.loc:@default_policy/sequential/action_1/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:
*N
shared_name?=default_policy/default_policy/sequential/action_1/kernel/Adam

^default_policy/default_policy/sequential/action_1/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp=default_policy/default_policy/sequential/action_1/kernel/Adam*<
_class2
0.loc:@default_policy/sequential/action_1/kernel*
_output_shapes
: 
Г
Ddefault_policy/default_policy/sequential/action_1/kernel/Adam/AssignAssignVariableOp=default_policy/default_policy/sequential/action_1/kernel/AdamOdefault_policy/default_policy/sequential/action_1/kernel/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Qdefault_policy/default_policy/sequential/action_1/kernel/Adam/Read/ReadVariableOpReadVariableOp=default_policy/default_policy/sequential/action_1/kernel/Adam*<
_class2
0.loc:@default_policy/sequential/action_1/kernel* 
_output_shapes
:
*
dtype0
№
adefault_policy/default_policy/sequential/action_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*<
_class2
0.loc:@default_policy/sequential/action_1/kernel*
_output_shapes
:*
dtype0*
valueB"     
к
Wdefault_policy/default_policy/sequential/action_1/kernel/Adam_1/Initializer/zeros/ConstConst*<
_class2
0.loc:@default_policy/sequential/action_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *    

Qdefault_policy/default_policy/sequential/action_1/kernel/Adam_1/Initializer/zerosFilladefault_policy/default_policy/sequential/action_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorWdefault_policy/default_policy/sequential/action_1/kernel/Adam_1/Initializer/zeros/Const*
T0*<
_class2
0.loc:@default_policy/sequential/action_1/kernel* 
_output_shapes
:
*

index_type0
Т
?default_policy/default_policy/sequential/action_1/kernel/Adam_1VarHandleOp*<
_class2
0.loc:@default_policy/sequential/action_1/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:
*P
shared_nameA?default_policy/default_policy/sequential/action_1/kernel/Adam_1

`default_policy/default_policy/sequential/action_1/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp?default_policy/default_policy/sequential/action_1/kernel/Adam_1*<
_class2
0.loc:@default_policy/sequential/action_1/kernel*
_output_shapes
: 
Й
Fdefault_policy/default_policy/sequential/action_1/kernel/Adam_1/AssignAssignVariableOp?default_policy/default_policy/sequential/action_1/kernel/Adam_1Qdefault_policy/default_policy/sequential/action_1/kernel/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Sdefault_policy/default_policy/sequential/action_1/kernel/Adam_1/Read/ReadVariableOpReadVariableOp?default_policy/default_policy/sequential/action_1/kernel/Adam_1*<
_class2
0.loc:@default_policy/sequential/action_1/kernel* 
_output_shapes
:
*
dtype0
и
Mdefault_policy/default_policy/sequential/action_1/bias/Adam/Initializer/zerosConst*:
_class0
.,loc:@default_policy/sequential/action_1/bias*
_output_shapes	
:*
dtype0*
valueB*    
Г
;default_policy/default_policy/sequential/action_1/bias/AdamVarHandleOp*:
_class0
.,loc:@default_policy/sequential/action_1/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*L
shared_name=;default_policy/default_policy/sequential/action_1/bias/Adam

\default_policy/default_policy/sequential/action_1/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp;default_policy/default_policy/sequential/action_1/bias/Adam*:
_class0
.,loc:@default_policy/sequential/action_1/bias*
_output_shapes
: 
­
Bdefault_policy/default_policy/sequential/action_1/bias/Adam/AssignAssignVariableOp;default_policy/default_policy/sequential/action_1/bias/AdamMdefault_policy/default_policy/sequential/action_1/bias/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Odefault_policy/default_policy/sequential/action_1/bias/Adam/Read/ReadVariableOpReadVariableOp;default_policy/default_policy/sequential/action_1/bias/Adam*:
_class0
.,loc:@default_policy/sequential/action_1/bias*
_output_shapes	
:*
dtype0
к
Odefault_policy/default_policy/sequential/action_1/bias/Adam_1/Initializer/zerosConst*:
_class0
.,loc:@default_policy/sequential/action_1/bias*
_output_shapes	
:*
dtype0*
valueB*    
З
=default_policy/default_policy/sequential/action_1/bias/Adam_1VarHandleOp*:
_class0
.,loc:@default_policy/sequential/action_1/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*N
shared_name?=default_policy/default_policy/sequential/action_1/bias/Adam_1

^default_policy/default_policy/sequential/action_1/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp=default_policy/default_policy/sequential/action_1/bias/Adam_1*:
_class0
.,loc:@default_policy/sequential/action_1/bias*
_output_shapes
: 
Г
Ddefault_policy/default_policy/sequential/action_1/bias/Adam_1/AssignAssignVariableOp=default_policy/default_policy/sequential/action_1/bias/Adam_1Odefault_policy/default_policy/sequential/action_1/bias/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Qdefault_policy/default_policy/sequential/action_1/bias/Adam_1/Read/ReadVariableOpReadVariableOp=default_policy/default_policy/sequential/action_1/bias/Adam_1*:
_class0
.,loc:@default_policy/sequential/action_1/bias*
_output_shapes	
:*
dtype0
ю
_default_policy/default_policy/sequential/action_2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*<
_class2
0.loc:@default_policy/sequential/action_2/kernel*
_output_shapes
:*
dtype0*
valueB"      
и
Udefault_policy/default_policy/sequential/action_2/kernel/Adam/Initializer/zeros/ConstConst*<
_class2
0.loc:@default_policy/sequential/action_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *    

Odefault_policy/default_policy/sequential/action_2/kernel/Adam/Initializer/zerosFill_default_policy/default_policy/sequential/action_2/kernel/Adam/Initializer/zeros/shape_as_tensorUdefault_policy/default_policy/sequential/action_2/kernel/Adam/Initializer/zeros/Const*
T0*<
_class2
0.loc:@default_policy/sequential/action_2/kernel* 
_output_shapes
:
*

index_type0
О
=default_policy/default_policy/sequential/action_2/kernel/AdamVarHandleOp*<
_class2
0.loc:@default_policy/sequential/action_2/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:
*N
shared_name?=default_policy/default_policy/sequential/action_2/kernel/Adam

^default_policy/default_policy/sequential/action_2/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp=default_policy/default_policy/sequential/action_2/kernel/Adam*<
_class2
0.loc:@default_policy/sequential/action_2/kernel*
_output_shapes
: 
Г
Ddefault_policy/default_policy/sequential/action_2/kernel/Adam/AssignAssignVariableOp=default_policy/default_policy/sequential/action_2/kernel/AdamOdefault_policy/default_policy/sequential/action_2/kernel/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Qdefault_policy/default_policy/sequential/action_2/kernel/Adam/Read/ReadVariableOpReadVariableOp=default_policy/default_policy/sequential/action_2/kernel/Adam*<
_class2
0.loc:@default_policy/sequential/action_2/kernel* 
_output_shapes
:
*
dtype0
№
adefault_policy/default_policy/sequential/action_2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*<
_class2
0.loc:@default_policy/sequential/action_2/kernel*
_output_shapes
:*
dtype0*
valueB"      
к
Wdefault_policy/default_policy/sequential/action_2/kernel/Adam_1/Initializer/zeros/ConstConst*<
_class2
0.loc:@default_policy/sequential/action_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *    

Qdefault_policy/default_policy/sequential/action_2/kernel/Adam_1/Initializer/zerosFilladefault_policy/default_policy/sequential/action_2/kernel/Adam_1/Initializer/zeros/shape_as_tensorWdefault_policy/default_policy/sequential/action_2/kernel/Adam_1/Initializer/zeros/Const*
T0*<
_class2
0.loc:@default_policy/sequential/action_2/kernel* 
_output_shapes
:
*

index_type0
Т
?default_policy/default_policy/sequential/action_2/kernel/Adam_1VarHandleOp*<
_class2
0.loc:@default_policy/sequential/action_2/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:
*P
shared_nameA?default_policy/default_policy/sequential/action_2/kernel/Adam_1

`default_policy/default_policy/sequential/action_2/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp?default_policy/default_policy/sequential/action_2/kernel/Adam_1*<
_class2
0.loc:@default_policy/sequential/action_2/kernel*
_output_shapes
: 
Й
Fdefault_policy/default_policy/sequential/action_2/kernel/Adam_1/AssignAssignVariableOp?default_policy/default_policy/sequential/action_2/kernel/Adam_1Qdefault_policy/default_policy/sequential/action_2/kernel/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Sdefault_policy/default_policy/sequential/action_2/kernel/Adam_1/Read/ReadVariableOpReadVariableOp?default_policy/default_policy/sequential/action_2/kernel/Adam_1*<
_class2
0.loc:@default_policy/sequential/action_2/kernel* 
_output_shapes
:
*
dtype0
и
Mdefault_policy/default_policy/sequential/action_2/bias/Adam/Initializer/zerosConst*:
_class0
.,loc:@default_policy/sequential/action_2/bias*
_output_shapes	
:*
dtype0*
valueB*    
Г
;default_policy/default_policy/sequential/action_2/bias/AdamVarHandleOp*:
_class0
.,loc:@default_policy/sequential/action_2/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*L
shared_name=;default_policy/default_policy/sequential/action_2/bias/Adam

\default_policy/default_policy/sequential/action_2/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp;default_policy/default_policy/sequential/action_2/bias/Adam*:
_class0
.,loc:@default_policy/sequential/action_2/bias*
_output_shapes
: 
­
Bdefault_policy/default_policy/sequential/action_2/bias/Adam/AssignAssignVariableOp;default_policy/default_policy/sequential/action_2/bias/AdamMdefault_policy/default_policy/sequential/action_2/bias/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Odefault_policy/default_policy/sequential/action_2/bias/Adam/Read/ReadVariableOpReadVariableOp;default_policy/default_policy/sequential/action_2/bias/Adam*:
_class0
.,loc:@default_policy/sequential/action_2/bias*
_output_shapes	
:*
dtype0
к
Odefault_policy/default_policy/sequential/action_2/bias/Adam_1/Initializer/zerosConst*:
_class0
.,loc:@default_policy/sequential/action_2/bias*
_output_shapes	
:*
dtype0*
valueB*    
З
=default_policy/default_policy/sequential/action_2/bias/Adam_1VarHandleOp*:
_class0
.,loc:@default_policy/sequential/action_2/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*N
shared_name?=default_policy/default_policy/sequential/action_2/bias/Adam_1

^default_policy/default_policy/sequential/action_2/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp=default_policy/default_policy/sequential/action_2/bias/Adam_1*:
_class0
.,loc:@default_policy/sequential/action_2/bias*
_output_shapes
: 
Г
Ddefault_policy/default_policy/sequential/action_2/bias/Adam_1/AssignAssignVariableOp=default_policy/default_policy/sequential/action_2/bias/Adam_1Odefault_policy/default_policy/sequential/action_2/bias/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Qdefault_policy/default_policy/sequential/action_2/bias/Adam_1/Read/ReadVariableOpReadVariableOp=default_policy/default_policy/sequential/action_2/bias/Adam_1*:
_class0
.,loc:@default_policy/sequential/action_2/bias*
_output_shapes	
:*
dtype0
ђ
adefault_policy/default_policy/sequential/action_out/kernel/Adam/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@default_policy/sequential/action_out/kernel*
_output_shapes
:*
dtype0*
valueB"      
м
Wdefault_policy/default_policy/sequential/action_out/kernel/Adam/Initializer/zeros/ConstConst*>
_class4
20loc:@default_policy/sequential/action_out/kernel*
_output_shapes
: *
dtype0*
valueB
 *    

Qdefault_policy/default_policy/sequential/action_out/kernel/Adam/Initializer/zerosFilladefault_policy/default_policy/sequential/action_out/kernel/Adam/Initializer/zeros/shape_as_tensorWdefault_policy/default_policy/sequential/action_out/kernel/Adam/Initializer/zeros/Const*
T0*>
_class4
20loc:@default_policy/sequential/action_out/kernel*
_output_shapes
:	*

index_type0
У
?default_policy/default_policy/sequential/action_out/kernel/AdamVarHandleOp*>
_class4
20loc:@default_policy/sequential/action_out/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:	*P
shared_nameA?default_policy/default_policy/sequential/action_out/kernel/Adam

`default_policy/default_policy/sequential/action_out/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp?default_policy/default_policy/sequential/action_out/kernel/Adam*>
_class4
20loc:@default_policy/sequential/action_out/kernel*
_output_shapes
: 
Й
Fdefault_policy/default_policy/sequential/action_out/kernel/Adam/AssignAssignVariableOp?default_policy/default_policy/sequential/action_out/kernel/AdamQdefault_policy/default_policy/sequential/action_out/kernel/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Sdefault_policy/default_policy/sequential/action_out/kernel/Adam/Read/ReadVariableOpReadVariableOp?default_policy/default_policy/sequential/action_out/kernel/Adam*>
_class4
20loc:@default_policy/sequential/action_out/kernel*
_output_shapes
:	*
dtype0
є
cdefault_policy/default_policy/sequential/action_out/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@default_policy/sequential/action_out/kernel*
_output_shapes
:*
dtype0*
valueB"      
о
Ydefault_policy/default_policy/sequential/action_out/kernel/Adam_1/Initializer/zeros/ConstConst*>
_class4
20loc:@default_policy/sequential/action_out/kernel*
_output_shapes
: *
dtype0*
valueB
 *    

Sdefault_policy/default_policy/sequential/action_out/kernel/Adam_1/Initializer/zerosFillcdefault_policy/default_policy/sequential/action_out/kernel/Adam_1/Initializer/zeros/shape_as_tensorYdefault_policy/default_policy/sequential/action_out/kernel/Adam_1/Initializer/zeros/Const*
T0*>
_class4
20loc:@default_policy/sequential/action_out/kernel*
_output_shapes
:	*

index_type0
Ч
Adefault_policy/default_policy/sequential/action_out/kernel/Adam_1VarHandleOp*>
_class4
20loc:@default_policy/sequential/action_out/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:	*R
shared_nameCAdefault_policy/default_policy/sequential/action_out/kernel/Adam_1

bdefault_policy/default_policy/sequential/action_out/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdefault_policy/default_policy/sequential/action_out/kernel/Adam_1*>
_class4
20loc:@default_policy/sequential/action_out/kernel*
_output_shapes
: 
П
Hdefault_policy/default_policy/sequential/action_out/kernel/Adam_1/AssignAssignVariableOpAdefault_policy/default_policy/sequential/action_out/kernel/Adam_1Sdefault_policy/default_policy/sequential/action_out/kernel/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Udefault_policy/default_policy/sequential/action_out/kernel/Adam_1/Read/ReadVariableOpReadVariableOpAdefault_policy/default_policy/sequential/action_out/kernel/Adam_1*>
_class4
20loc:@default_policy/sequential/action_out/kernel*
_output_shapes
:	*
dtype0
к
Odefault_policy/default_policy/sequential/action_out/bias/Adam/Initializer/zerosConst*<
_class2
0.loc:@default_policy/sequential/action_out/bias*
_output_shapes
:*
dtype0*
valueB*    
И
=default_policy/default_policy/sequential/action_out/bias/AdamVarHandleOp*<
_class2
0.loc:@default_policy/sequential/action_out/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*N
shared_name?=default_policy/default_policy/sequential/action_out/bias/Adam

^default_policy/default_policy/sequential/action_out/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp=default_policy/default_policy/sequential/action_out/bias/Adam*<
_class2
0.loc:@default_policy/sequential/action_out/bias*
_output_shapes
: 
Г
Ddefault_policy/default_policy/sequential/action_out/bias/Adam/AssignAssignVariableOp=default_policy/default_policy/sequential/action_out/bias/AdamOdefault_policy/default_policy/sequential/action_out/bias/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Qdefault_policy/default_policy/sequential/action_out/bias/Adam/Read/ReadVariableOpReadVariableOp=default_policy/default_policy/sequential/action_out/bias/Adam*<
_class2
0.loc:@default_policy/sequential/action_out/bias*
_output_shapes
:*
dtype0
м
Qdefault_policy/default_policy/sequential/action_out/bias/Adam_1/Initializer/zerosConst*<
_class2
0.loc:@default_policy/sequential/action_out/bias*
_output_shapes
:*
dtype0*
valueB*    
М
?default_policy/default_policy/sequential/action_out/bias/Adam_1VarHandleOp*<
_class2
0.loc:@default_policy/sequential/action_out/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*P
shared_nameA?default_policy/default_policy/sequential/action_out/bias/Adam_1

`default_policy/default_policy/sequential/action_out/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp?default_policy/default_policy/sequential/action_out/bias/Adam_1*<
_class2
0.loc:@default_policy/sequential/action_out/bias*
_output_shapes
: 
Й
Fdefault_policy/default_policy/sequential/action_out/bias/Adam_1/AssignAssignVariableOp?default_policy/default_policy/sequential/action_out/bias/Adam_1Qdefault_policy/default_policy/sequential/action_out/bias/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Sdefault_policy/default_policy/sequential/action_out/bias/Adam_1/Read/ReadVariableOpReadVariableOp?default_policy/default_policy/sequential/action_out/bias/Adam_1*<
_class2
0.loc:@default_policy/sequential/action_out/bias*
_output_shapes
:*
dtype0
f
!default_policy/Adam/learning_rateConst*
_output_shapes
: *
dtype0*
valueB
 *Зб8
^
default_policy/Adam/beta1Const*
_output_shapes
: *
dtype0*
valueB
 *fff?
^
default_policy/Adam/beta2Const*
_output_shapes
: *
dtype0*
valueB
 *wО?
`
default_policy/Adam/epsilonConst*
_output_shapes
: *
dtype0*
valueB
 *wЬ+2
И
edefault_policy/Adam/update_default_policy/sequential/action_1/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power*
_output_shapes
: *
dtype0
К
gdefault_policy/Adam/update_default_policy/sequential/action_1/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power*
_output_shapes
: *
dtype0

Vdefault_policy/Adam/update_default_policy/sequential/action_1/kernel/ResourceApplyAdamResourceApplyAdam)default_policy/sequential/action_1/kernel=default_policy/default_policy/sequential/action_1/kernel/Adam?default_policy/default_policy/sequential/action_1/kernel/Adam_1edefault_policy/Adam/update_default_policy/sequential/action_1/kernel/ResourceApplyAdam/ReadVariableOpgdefault_policy/Adam/update_default_policy/sequential/action_1/kernel/ResourceApplyAdam/ReadVariableOp_1!default_policy/Adam/learning_ratedefault_policy/Adam/beta1default_policy/Adam/beta2default_policy/Adam/epsilondefault_policy/Identity*
T0*<
_class2
0.loc:@default_policy/sequential/action_1/kernel*&
 _has_manual_control_dependencies(*
use_locking( *
use_nesterov( 
Ж
cdefault_policy/Adam/update_default_policy/sequential/action_1/bias/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power*
_output_shapes
: *
dtype0
И
edefault_policy/Adam/update_default_policy/sequential/action_1/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power*
_output_shapes
: *
dtype0
џ
Tdefault_policy/Adam/update_default_policy/sequential/action_1/bias/ResourceApplyAdamResourceApplyAdam'default_policy/sequential/action_1/bias;default_policy/default_policy/sequential/action_1/bias/Adam=default_policy/default_policy/sequential/action_1/bias/Adam_1cdefault_policy/Adam/update_default_policy/sequential/action_1/bias/ResourceApplyAdam/ReadVariableOpedefault_policy/Adam/update_default_policy/sequential/action_1/bias/ResourceApplyAdam/ReadVariableOp_1!default_policy/Adam/learning_ratedefault_policy/Adam/beta1default_policy/Adam/beta2default_policy/Adam/epsilondefault_policy/Identity_1*
T0*:
_class0
.,loc:@default_policy/sequential/action_1/bias*&
 _has_manual_control_dependencies(*
use_locking( *
use_nesterov( 
И
edefault_policy/Adam/update_default_policy/sequential/action_2/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power*
_output_shapes
: *
dtype0
К
gdefault_policy/Adam/update_default_policy/sequential/action_2/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power*
_output_shapes
: *
dtype0

Vdefault_policy/Adam/update_default_policy/sequential/action_2/kernel/ResourceApplyAdamResourceApplyAdam)default_policy/sequential/action_2/kernel=default_policy/default_policy/sequential/action_2/kernel/Adam?default_policy/default_policy/sequential/action_2/kernel/Adam_1edefault_policy/Adam/update_default_policy/sequential/action_2/kernel/ResourceApplyAdam/ReadVariableOpgdefault_policy/Adam/update_default_policy/sequential/action_2/kernel/ResourceApplyAdam/ReadVariableOp_1!default_policy/Adam/learning_ratedefault_policy/Adam/beta1default_policy/Adam/beta2default_policy/Adam/epsilondefault_policy/Identity_2*
T0*<
_class2
0.loc:@default_policy/sequential/action_2/kernel*&
 _has_manual_control_dependencies(*
use_locking( *
use_nesterov( 
Ж
cdefault_policy/Adam/update_default_policy/sequential/action_2/bias/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power*
_output_shapes
: *
dtype0
И
edefault_policy/Adam/update_default_policy/sequential/action_2/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power*
_output_shapes
: *
dtype0
џ
Tdefault_policy/Adam/update_default_policy/sequential/action_2/bias/ResourceApplyAdamResourceApplyAdam'default_policy/sequential/action_2/bias;default_policy/default_policy/sequential/action_2/bias/Adam=default_policy/default_policy/sequential/action_2/bias/Adam_1cdefault_policy/Adam/update_default_policy/sequential/action_2/bias/ResourceApplyAdam/ReadVariableOpedefault_policy/Adam/update_default_policy/sequential/action_2/bias/ResourceApplyAdam/ReadVariableOp_1!default_policy/Adam/learning_ratedefault_policy/Adam/beta1default_policy/Adam/beta2default_policy/Adam/epsilondefault_policy/Identity_3*
T0*:
_class0
.,loc:@default_policy/sequential/action_2/bias*&
 _has_manual_control_dependencies(*
use_locking( *
use_nesterov( 
К
gdefault_policy/Adam/update_default_policy/sequential/action_out/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power*
_output_shapes
: *
dtype0
М
idefault_policy/Adam/update_default_policy/sequential/action_out/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power*
_output_shapes
: *
dtype0

Xdefault_policy/Adam/update_default_policy/sequential/action_out/kernel/ResourceApplyAdamResourceApplyAdam+default_policy/sequential/action_out/kernel?default_policy/default_policy/sequential/action_out/kernel/AdamAdefault_policy/default_policy/sequential/action_out/kernel/Adam_1gdefault_policy/Adam/update_default_policy/sequential/action_out/kernel/ResourceApplyAdam/ReadVariableOpidefault_policy/Adam/update_default_policy/sequential/action_out/kernel/ResourceApplyAdam/ReadVariableOp_1!default_policy/Adam/learning_ratedefault_policy/Adam/beta1default_policy/Adam/beta2default_policy/Adam/epsilondefault_policy/Identity_4*
T0*>
_class4
20loc:@default_policy/sequential/action_out/kernel*&
 _has_manual_control_dependencies(*
use_locking( *
use_nesterov( 
И
edefault_policy/Adam/update_default_policy/sequential/action_out/bias/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power*
_output_shapes
: *
dtype0
К
gdefault_policy/Adam/update_default_policy/sequential/action_out/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power*
_output_shapes
: *
dtype0

Vdefault_policy/Adam/update_default_policy/sequential/action_out/bias/ResourceApplyAdamResourceApplyAdam)default_policy/sequential/action_out/bias=default_policy/default_policy/sequential/action_out/bias/Adam?default_policy/default_policy/sequential/action_out/bias/Adam_1edefault_policy/Adam/update_default_policy/sequential/action_out/bias/ResourceApplyAdam/ReadVariableOpgdefault_policy/Adam/update_default_policy/sequential/action_out/bias/ResourceApplyAdam/ReadVariableOp_1!default_policy/Adam/learning_ratedefault_policy/Adam/beta1default_policy/Adam/beta2default_policy/Adam/epsilondefault_policy/Identity_5*
T0*<
_class2
0.loc:@default_policy/sequential/action_out/bias*&
 _has_manual_control_dependencies(*
use_locking( *
use_nesterov( 

"default_policy/Adam/ReadVariableOpReadVariableOpdefault_policy/beta1_powerU^default_policy/Adam/update_default_policy/sequential/action_1/bias/ResourceApplyAdamW^default_policy/Adam/update_default_policy/sequential/action_1/kernel/ResourceApplyAdamU^default_policy/Adam/update_default_policy/sequential/action_2/bias/ResourceApplyAdamW^default_policy/Adam/update_default_policy/sequential/action_2/kernel/ResourceApplyAdamW^default_policy/Adam/update_default_policy/sequential/action_out/bias/ResourceApplyAdamY^default_policy/Adam/update_default_policy/sequential/action_out/kernel/ResourceApplyAdam*
_output_shapes
: *
dtype0
К
default_policy/Adam/mulMul"default_policy/Adam/ReadVariableOpdefault_policy/Adam/beta1*
T0*:
_class0
.,loc:@default_policy/sequential/action_1/bias*
_output_shapes
: 
є
$default_policy/Adam/AssignVariableOpAssignVariableOpdefault_policy/beta1_powerdefault_policy/Adam/mul*:
_class0
.,loc:@default_policy/sequential/action_1/bias*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
ю
$default_policy/Adam/ReadVariableOp_1ReadVariableOpdefault_policy/beta1_power%^default_policy/Adam/AssignVariableOpU^default_policy/Adam/update_default_policy/sequential/action_1/bias/ResourceApplyAdamW^default_policy/Adam/update_default_policy/sequential/action_1/kernel/ResourceApplyAdamU^default_policy/Adam/update_default_policy/sequential/action_2/bias/ResourceApplyAdamW^default_policy/Adam/update_default_policy/sequential/action_2/kernel/ResourceApplyAdamW^default_policy/Adam/update_default_policy/sequential/action_out/bias/ResourceApplyAdamY^default_policy/Adam/update_default_policy/sequential/action_out/kernel/ResourceApplyAdam*:
_class0
.,loc:@default_policy/sequential/action_1/bias*
_output_shapes
: *
dtype0

$default_policy/Adam/ReadVariableOp_2ReadVariableOpdefault_policy/beta2_powerU^default_policy/Adam/update_default_policy/sequential/action_1/bias/ResourceApplyAdamW^default_policy/Adam/update_default_policy/sequential/action_1/kernel/ResourceApplyAdamU^default_policy/Adam/update_default_policy/sequential/action_2/bias/ResourceApplyAdamW^default_policy/Adam/update_default_policy/sequential/action_2/kernel/ResourceApplyAdamW^default_policy/Adam/update_default_policy/sequential/action_out/bias/ResourceApplyAdamY^default_policy/Adam/update_default_policy/sequential/action_out/kernel/ResourceApplyAdam*
_output_shapes
: *
dtype0
О
default_policy/Adam/mul_1Mul$default_policy/Adam/ReadVariableOp_2default_policy/Adam/beta2*
T0*:
_class0
.,loc:@default_policy/sequential/action_1/bias*
_output_shapes
: 
ј
&default_policy/Adam/AssignVariableOp_1AssignVariableOpdefault_policy/beta2_powerdefault_policy/Adam/mul_1*:
_class0
.,loc:@default_policy/sequential/action_1/bias*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
№
$default_policy/Adam/ReadVariableOp_3ReadVariableOpdefault_policy/beta2_power'^default_policy/Adam/AssignVariableOp_1U^default_policy/Adam/update_default_policy/sequential/action_1/bias/ResourceApplyAdamW^default_policy/Adam/update_default_policy/sequential/action_1/kernel/ResourceApplyAdamU^default_policy/Adam/update_default_policy/sequential/action_2/bias/ResourceApplyAdamW^default_policy/Adam/update_default_policy/sequential/action_2/kernel/ResourceApplyAdamW^default_policy/Adam/update_default_policy/sequential/action_out/bias/ResourceApplyAdamY^default_policy/Adam/update_default_policy/sequential/action_out/kernel/ResourceApplyAdam*:
_class0
.,loc:@default_policy/sequential/action_1/bias*
_output_shapes
: *
dtype0
Ї
default_policy/AdamNoOp%^default_policy/Adam/AssignVariableOp'^default_policy/Adam/AssignVariableOp_1U^default_policy/Adam/update_default_policy/sequential/action_1/bias/ResourceApplyAdamW^default_policy/Adam/update_default_policy/sequential/action_1/kernel/ResourceApplyAdamU^default_policy/Adam/update_default_policy/sequential/action_2/bias/ResourceApplyAdamW^default_policy/Adam/update_default_policy/sequential/action_2/kernel/ResourceApplyAdamW^default_policy/Adam/update_default_policy/sequential/action_out/bias/ResourceApplyAdamY^default_policy/Adam/update_default_policy/sequential/action_out/kernel/ResourceApplyAdam*&
 _has_manual_control_dependencies(
Л
6default_policy/beta1_power_1/Initializer/initial_valueConst*>
_class4
20loc:@default_policy/sequential_1/q_hidden_0/bias*
_output_shapes
: *
dtype0*
valueB
 *fff?
є
default_policy/beta1_power_1VarHandleOp*>
_class4
20loc:@default_policy/sequential_1/q_hidden_0/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *-
shared_namedefault_policy/beta1_power_1
Щ
=default_policy/beta1_power_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/beta1_power_1*>
_class4
20loc:@default_policy/sequential_1/q_hidden_0/bias*
_output_shapes
: 
и
#default_policy/beta1_power_1/AssignAssignVariableOpdefault_policy/beta1_power_16default_policy/beta1_power_1/Initializer/initial_value*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Х
0default_policy/beta1_power_1/Read/ReadVariableOpReadVariableOpdefault_policy/beta1_power_1*>
_class4
20loc:@default_policy/sequential_1/q_hidden_0/bias*
_output_shapes
: *
dtype0
Л
6default_policy/beta2_power_1/Initializer/initial_valueConst*>
_class4
20loc:@default_policy/sequential_1/q_hidden_0/bias*
_output_shapes
: *
dtype0*
valueB
 *wО?
є
default_policy/beta2_power_1VarHandleOp*>
_class4
20loc:@default_policy/sequential_1/q_hidden_0/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *-
shared_namedefault_policy/beta2_power_1
Щ
=default_policy/beta2_power_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/beta2_power_1*>
_class4
20loc:@default_policy/sequential_1/q_hidden_0/bias*
_output_shapes
: 
и
#default_policy/beta2_power_1/AssignAssignVariableOpdefault_policy/beta2_power_16default_policy/beta2_power_1/Initializer/initial_value*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Х
0default_policy/beta2_power_1/Read/ReadVariableOpReadVariableOpdefault_policy/beta2_power_1*>
_class4
20loc:@default_policy/sequential_1/q_hidden_0/bias*
_output_shapes
: *
dtype0
і
cdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@default_policy/sequential_1/q_hidden_0/kernel*
_output_shapes
:*
dtype0*
valueB"     
р
Ydefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam/Initializer/zeros/ConstConst*@
_class6
42loc:@default_policy/sequential_1/q_hidden_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *    

Sdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam/Initializer/zerosFillcdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam/Initializer/zeros/shape_as_tensorYdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam/Initializer/zeros/Const*
T0*@
_class6
42loc:@default_policy/sequential_1/q_hidden_0/kernel* 
_output_shapes
:
*

index_type0
Ъ
Adefault_policy/default_policy/sequential_1/q_hidden_0/kernel/AdamVarHandleOp*@
_class6
42loc:@default_policy/sequential_1/q_hidden_0/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:
*R
shared_nameCAdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam

bdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam*@
_class6
42loc:@default_policy/sequential_1/q_hidden_0/kernel*
_output_shapes
: 
П
Hdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam/AssignAssignVariableOpAdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/AdamSdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Udefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam/Read/ReadVariableOpReadVariableOpAdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam*@
_class6
42loc:@default_policy/sequential_1/q_hidden_0/kernel* 
_output_shapes
:
*
dtype0
ј
edefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@default_policy/sequential_1/q_hidden_0/kernel*
_output_shapes
:*
dtype0*
valueB"     
т
[default_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam_1/Initializer/zeros/ConstConst*@
_class6
42loc:@default_policy/sequential_1/q_hidden_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
 
Udefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam_1/Initializer/zerosFilledefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam_1/Initializer/zeros/shape_as_tensor[default_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam_1/Initializer/zeros/Const*
T0*@
_class6
42loc:@default_policy/sequential_1/q_hidden_0/kernel* 
_output_shapes
:
*

index_type0
Ю
Cdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam_1VarHandleOp*@
_class6
42loc:@default_policy/sequential_1/q_hidden_0/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:
*T
shared_nameECdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam_1

ddefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpCdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam_1*@
_class6
42loc:@default_policy/sequential_1/q_hidden_0/kernel*
_output_shapes
: 
Х
Jdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam_1/AssignAssignVariableOpCdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam_1Udefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Wdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam_1/Read/ReadVariableOpReadVariableOpCdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam_1*@
_class6
42loc:@default_policy/sequential_1/q_hidden_0/kernel* 
_output_shapes
:
*
dtype0
р
Qdefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam/Initializer/zerosConst*>
_class4
20loc:@default_policy/sequential_1/q_hidden_0/bias*
_output_shapes	
:*
dtype0*
valueB*    
П
?default_policy/default_policy/sequential_1/q_hidden_0/bias/AdamVarHandleOp*>
_class4
20loc:@default_policy/sequential_1/q_hidden_0/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*P
shared_nameA?default_policy/default_policy/sequential_1/q_hidden_0/bias/Adam

`default_policy/default_policy/sequential_1/q_hidden_0/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp?default_policy/default_policy/sequential_1/q_hidden_0/bias/Adam*>
_class4
20loc:@default_policy/sequential_1/q_hidden_0/bias*
_output_shapes
: 
Й
Fdefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam/AssignAssignVariableOp?default_policy/default_policy/sequential_1/q_hidden_0/bias/AdamQdefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Sdefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam/Read/ReadVariableOpReadVariableOp?default_policy/default_policy/sequential_1/q_hidden_0/bias/Adam*>
_class4
20loc:@default_policy/sequential_1/q_hidden_0/bias*
_output_shapes	
:*
dtype0
т
Sdefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam_1/Initializer/zerosConst*>
_class4
20loc:@default_policy/sequential_1/q_hidden_0/bias*
_output_shapes	
:*
dtype0*
valueB*    
У
Adefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam_1VarHandleOp*>
_class4
20loc:@default_policy/sequential_1/q_hidden_0/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*R
shared_nameCAdefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam_1

bdefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam_1*>
_class4
20loc:@default_policy/sequential_1/q_hidden_0/bias*
_output_shapes
: 
П
Hdefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam_1/AssignAssignVariableOpAdefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam_1Sdefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Udefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam_1/Read/ReadVariableOpReadVariableOpAdefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam_1*>
_class4
20loc:@default_policy/sequential_1/q_hidden_0/bias*
_output_shapes	
:*
dtype0
і
cdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@default_policy/sequential_1/q_hidden_1/kernel*
_output_shapes
:*
dtype0*
valueB"      
р
Ydefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam/Initializer/zeros/ConstConst*@
_class6
42loc:@default_policy/sequential_1/q_hidden_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *    

Sdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam/Initializer/zerosFillcdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam/Initializer/zeros/shape_as_tensorYdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam/Initializer/zeros/Const*
T0*@
_class6
42loc:@default_policy/sequential_1/q_hidden_1/kernel* 
_output_shapes
:
*

index_type0
Ъ
Adefault_policy/default_policy/sequential_1/q_hidden_1/kernel/AdamVarHandleOp*@
_class6
42loc:@default_policy/sequential_1/q_hidden_1/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:
*R
shared_nameCAdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam

bdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam*@
_class6
42loc:@default_policy/sequential_1/q_hidden_1/kernel*
_output_shapes
: 
П
Hdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam/AssignAssignVariableOpAdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/AdamSdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Udefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam/Read/ReadVariableOpReadVariableOpAdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam*@
_class6
42loc:@default_policy/sequential_1/q_hidden_1/kernel* 
_output_shapes
:
*
dtype0
ј
edefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@default_policy/sequential_1/q_hidden_1/kernel*
_output_shapes
:*
dtype0*
valueB"      
т
[default_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam_1/Initializer/zeros/ConstConst*@
_class6
42loc:@default_policy/sequential_1/q_hidden_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
 
Udefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam_1/Initializer/zerosFilledefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor[default_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam_1/Initializer/zeros/Const*
T0*@
_class6
42loc:@default_policy/sequential_1/q_hidden_1/kernel* 
_output_shapes
:
*

index_type0
Ю
Cdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam_1VarHandleOp*@
_class6
42loc:@default_policy/sequential_1/q_hidden_1/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:
*T
shared_nameECdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam_1

ddefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpCdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam_1*@
_class6
42loc:@default_policy/sequential_1/q_hidden_1/kernel*
_output_shapes
: 
Х
Jdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam_1/AssignAssignVariableOpCdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam_1Udefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Wdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam_1/Read/ReadVariableOpReadVariableOpCdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam_1*@
_class6
42loc:@default_policy/sequential_1/q_hidden_1/kernel* 
_output_shapes
:
*
dtype0
р
Qdefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam/Initializer/zerosConst*>
_class4
20loc:@default_policy/sequential_1/q_hidden_1/bias*
_output_shapes	
:*
dtype0*
valueB*    
П
?default_policy/default_policy/sequential_1/q_hidden_1/bias/AdamVarHandleOp*>
_class4
20loc:@default_policy/sequential_1/q_hidden_1/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*P
shared_nameA?default_policy/default_policy/sequential_1/q_hidden_1/bias/Adam

`default_policy/default_policy/sequential_1/q_hidden_1/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp?default_policy/default_policy/sequential_1/q_hidden_1/bias/Adam*>
_class4
20loc:@default_policy/sequential_1/q_hidden_1/bias*
_output_shapes
: 
Й
Fdefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam/AssignAssignVariableOp?default_policy/default_policy/sequential_1/q_hidden_1/bias/AdamQdefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Sdefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam/Read/ReadVariableOpReadVariableOp?default_policy/default_policy/sequential_1/q_hidden_1/bias/Adam*>
_class4
20loc:@default_policy/sequential_1/q_hidden_1/bias*
_output_shapes	
:*
dtype0
т
Sdefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam_1/Initializer/zerosConst*>
_class4
20loc:@default_policy/sequential_1/q_hidden_1/bias*
_output_shapes	
:*
dtype0*
valueB*    
У
Adefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam_1VarHandleOp*>
_class4
20loc:@default_policy/sequential_1/q_hidden_1/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*R
shared_nameCAdefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam_1

bdefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam_1*>
_class4
20loc:@default_policy/sequential_1/q_hidden_1/bias*
_output_shapes
: 
П
Hdefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam_1/AssignAssignVariableOpAdefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam_1Sdefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Udefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam_1/Read/ReadVariableOpReadVariableOpAdefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam_1*>
_class4
20loc:@default_policy/sequential_1/q_hidden_1/bias*
_output_shapes	
:*
dtype0
т
Ndefault_policy/default_policy/sequential_1/q_out/kernel/Adam/Initializer/zerosConst*;
_class1
/-loc:@default_policy/sequential_1/q_out/kernel*
_output_shapes
:	*
dtype0*
valueB	*    
К
<default_policy/default_policy/sequential_1/q_out/kernel/AdamVarHandleOp*;
_class1
/-loc:@default_policy/sequential_1/q_out/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:	*M
shared_name><default_policy/default_policy/sequential_1/q_out/kernel/Adam

]default_policy/default_policy/sequential_1/q_out/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp<default_policy/default_policy/sequential_1/q_out/kernel/Adam*;
_class1
/-loc:@default_policy/sequential_1/q_out/kernel*
_output_shapes
: 
А
Cdefault_policy/default_policy/sequential_1/q_out/kernel/Adam/AssignAssignVariableOp<default_policy/default_policy/sequential_1/q_out/kernel/AdamNdefault_policy/default_policy/sequential_1/q_out/kernel/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Pdefault_policy/default_policy/sequential_1/q_out/kernel/Adam/Read/ReadVariableOpReadVariableOp<default_policy/default_policy/sequential_1/q_out/kernel/Adam*;
_class1
/-loc:@default_policy/sequential_1/q_out/kernel*
_output_shapes
:	*
dtype0
ф
Pdefault_policy/default_policy/sequential_1/q_out/kernel/Adam_1/Initializer/zerosConst*;
_class1
/-loc:@default_policy/sequential_1/q_out/kernel*
_output_shapes
:	*
dtype0*
valueB	*    
О
>default_policy/default_policy/sequential_1/q_out/kernel/Adam_1VarHandleOp*;
_class1
/-loc:@default_policy/sequential_1/q_out/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:	*O
shared_name@>default_policy/default_policy/sequential_1/q_out/kernel/Adam_1

_default_policy/default_policy/sequential_1/q_out/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp>default_policy/default_policy/sequential_1/q_out/kernel/Adam_1*;
_class1
/-loc:@default_policy/sequential_1/q_out/kernel*
_output_shapes
: 
Ж
Edefault_policy/default_policy/sequential_1/q_out/kernel/Adam_1/AssignAssignVariableOp>default_policy/default_policy/sequential_1/q_out/kernel/Adam_1Pdefault_policy/default_policy/sequential_1/q_out/kernel/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Rdefault_policy/default_policy/sequential_1/q_out/kernel/Adam_1/Read/ReadVariableOpReadVariableOp>default_policy/default_policy/sequential_1/q_out/kernel/Adam_1*;
_class1
/-loc:@default_policy/sequential_1/q_out/kernel*
_output_shapes
:	*
dtype0
д
Ldefault_policy/default_policy/sequential_1/q_out/bias/Adam/Initializer/zerosConst*9
_class/
-+loc:@default_policy/sequential_1/q_out/bias*
_output_shapes
:*
dtype0*
valueB*    
Џ
:default_policy/default_policy/sequential_1/q_out/bias/AdamVarHandleOp*9
_class/
-+loc:@default_policy/sequential_1/q_out/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*K
shared_name<:default_policy/default_policy/sequential_1/q_out/bias/Adam

[default_policy/default_policy/sequential_1/q_out/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp:default_policy/default_policy/sequential_1/q_out/bias/Adam*9
_class/
-+loc:@default_policy/sequential_1/q_out/bias*
_output_shapes
: 
Њ
Adefault_policy/default_policy/sequential_1/q_out/bias/Adam/AssignAssignVariableOp:default_policy/default_policy/sequential_1/q_out/bias/AdamLdefault_policy/default_policy/sequential_1/q_out/bias/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Ndefault_policy/default_policy/sequential_1/q_out/bias/Adam/Read/ReadVariableOpReadVariableOp:default_policy/default_policy/sequential_1/q_out/bias/Adam*9
_class/
-+loc:@default_policy/sequential_1/q_out/bias*
_output_shapes
:*
dtype0
ж
Ndefault_policy/default_policy/sequential_1/q_out/bias/Adam_1/Initializer/zerosConst*9
_class/
-+loc:@default_policy/sequential_1/q_out/bias*
_output_shapes
:*
dtype0*
valueB*    
Г
<default_policy/default_policy/sequential_1/q_out/bias/Adam_1VarHandleOp*9
_class/
-+loc:@default_policy/sequential_1/q_out/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*M
shared_name><default_policy/default_policy/sequential_1/q_out/bias/Adam_1

]default_policy/default_policy/sequential_1/q_out/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp<default_policy/default_policy/sequential_1/q_out/bias/Adam_1*9
_class/
-+loc:@default_policy/sequential_1/q_out/bias*
_output_shapes
: 
А
Cdefault_policy/default_policy/sequential_1/q_out/bias/Adam_1/AssignAssignVariableOp<default_policy/default_policy/sequential_1/q_out/bias/Adam_1Ndefault_policy/default_policy/sequential_1/q_out/bias/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Pdefault_policy/default_policy/sequential_1/q_out/bias/Adam_1/Read/ReadVariableOpReadVariableOp<default_policy/default_policy/sequential_1/q_out/bias/Adam_1*9
_class/
-+loc:@default_policy/sequential_1/q_out/bias*
_output_shapes
:*
dtype0
h
#default_policy/Adam_1/learning_rateConst*
_output_shapes
: *
dtype0*
valueB
 *Зб8
`
default_policy/Adam_1/beta1Const*
_output_shapes
: *
dtype0*
valueB
 *fff?
`
default_policy/Adam_1/beta2Const*
_output_shapes
: *
dtype0*
valueB
 *wО?
b
default_policy/Adam_1/epsilonConst*
_output_shapes
: *
dtype0*
valueB
 *wЬ+2
Р
kdefault_policy/Adam_1/update_default_policy/sequential_1/q_hidden_0/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power_1*
_output_shapes
: *
dtype0
Т
mdefault_policy/Adam_1/update_default_policy/sequential_1/q_hidden_0/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power_1*
_output_shapes
: *
dtype0
З
\default_policy/Adam_1/update_default_policy/sequential_1/q_hidden_0/kernel/ResourceApplyAdamResourceApplyAdam-default_policy/sequential_1/q_hidden_0/kernelAdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/AdamCdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam_1kdefault_policy/Adam_1/update_default_policy/sequential_1/q_hidden_0/kernel/ResourceApplyAdam/ReadVariableOpmdefault_policy/Adam_1/update_default_policy/sequential_1/q_hidden_0/kernel/ResourceApplyAdam/ReadVariableOp_1#default_policy/Adam_1/learning_ratedefault_policy/Adam_1/beta1default_policy/Adam_1/beta2default_policy/Adam_1/epsilondefault_policy/Identity_6*
T0*@
_class6
42loc:@default_policy/sequential_1/q_hidden_0/kernel*&
 _has_manual_control_dependencies(*
use_locking( *
use_nesterov( 
О
idefault_policy/Adam_1/update_default_policy/sequential_1/q_hidden_0/bias/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power_1*
_output_shapes
: *
dtype0
Р
kdefault_policy/Adam_1/update_default_policy/sequential_1/q_hidden_0/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power_1*
_output_shapes
: *
dtype0
Љ
Zdefault_policy/Adam_1/update_default_policy/sequential_1/q_hidden_0/bias/ResourceApplyAdamResourceApplyAdam+default_policy/sequential_1/q_hidden_0/bias?default_policy/default_policy/sequential_1/q_hidden_0/bias/AdamAdefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam_1idefault_policy/Adam_1/update_default_policy/sequential_1/q_hidden_0/bias/ResourceApplyAdam/ReadVariableOpkdefault_policy/Adam_1/update_default_policy/sequential_1/q_hidden_0/bias/ResourceApplyAdam/ReadVariableOp_1#default_policy/Adam_1/learning_ratedefault_policy/Adam_1/beta1default_policy/Adam_1/beta2default_policy/Adam_1/epsilondefault_policy/Identity_7*
T0*>
_class4
20loc:@default_policy/sequential_1/q_hidden_0/bias*&
 _has_manual_control_dependencies(*
use_locking( *
use_nesterov( 
Р
kdefault_policy/Adam_1/update_default_policy/sequential_1/q_hidden_1/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power_1*
_output_shapes
: *
dtype0
Т
mdefault_policy/Adam_1/update_default_policy/sequential_1/q_hidden_1/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power_1*
_output_shapes
: *
dtype0
З
\default_policy/Adam_1/update_default_policy/sequential_1/q_hidden_1/kernel/ResourceApplyAdamResourceApplyAdam-default_policy/sequential_1/q_hidden_1/kernelAdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/AdamCdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam_1kdefault_policy/Adam_1/update_default_policy/sequential_1/q_hidden_1/kernel/ResourceApplyAdam/ReadVariableOpmdefault_policy/Adam_1/update_default_policy/sequential_1/q_hidden_1/kernel/ResourceApplyAdam/ReadVariableOp_1#default_policy/Adam_1/learning_ratedefault_policy/Adam_1/beta1default_policy/Adam_1/beta2default_policy/Adam_1/epsilondefault_policy/Identity_8*
T0*@
_class6
42loc:@default_policy/sequential_1/q_hidden_1/kernel*&
 _has_manual_control_dependencies(*
use_locking( *
use_nesterov( 
О
idefault_policy/Adam_1/update_default_policy/sequential_1/q_hidden_1/bias/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power_1*
_output_shapes
: *
dtype0
Р
kdefault_policy/Adam_1/update_default_policy/sequential_1/q_hidden_1/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power_1*
_output_shapes
: *
dtype0
Љ
Zdefault_policy/Adam_1/update_default_policy/sequential_1/q_hidden_1/bias/ResourceApplyAdamResourceApplyAdam+default_policy/sequential_1/q_hidden_1/bias?default_policy/default_policy/sequential_1/q_hidden_1/bias/AdamAdefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam_1idefault_policy/Adam_1/update_default_policy/sequential_1/q_hidden_1/bias/ResourceApplyAdam/ReadVariableOpkdefault_policy/Adam_1/update_default_policy/sequential_1/q_hidden_1/bias/ResourceApplyAdam/ReadVariableOp_1#default_policy/Adam_1/learning_ratedefault_policy/Adam_1/beta1default_policy/Adam_1/beta2default_policy/Adam_1/epsilondefault_policy/Identity_9*
T0*>
_class4
20loc:@default_policy/sequential_1/q_hidden_1/bias*&
 _has_manual_control_dependencies(*
use_locking( *
use_nesterov( 
Л
fdefault_policy/Adam_1/update_default_policy/sequential_1/q_out/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power_1*
_output_shapes
: *
dtype0
Н
hdefault_policy/Adam_1/update_default_policy/sequential_1/q_out/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power_1*
_output_shapes
: *
dtype0

Wdefault_policy/Adam_1/update_default_policy/sequential_1/q_out/kernel/ResourceApplyAdamResourceApplyAdam(default_policy/sequential_1/q_out/kernel<default_policy/default_policy/sequential_1/q_out/kernel/Adam>default_policy/default_policy/sequential_1/q_out/kernel/Adam_1fdefault_policy/Adam_1/update_default_policy/sequential_1/q_out/kernel/ResourceApplyAdam/ReadVariableOphdefault_policy/Adam_1/update_default_policy/sequential_1/q_out/kernel/ResourceApplyAdam/ReadVariableOp_1#default_policy/Adam_1/learning_ratedefault_policy/Adam_1/beta1default_policy/Adam_1/beta2default_policy/Adam_1/epsilondefault_policy/Identity_10*
T0*;
_class1
/-loc:@default_policy/sequential_1/q_out/kernel*&
 _has_manual_control_dependencies(*
use_locking( *
use_nesterov( 
Й
ddefault_policy/Adam_1/update_default_policy/sequential_1/q_out/bias/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power_1*
_output_shapes
: *
dtype0
Л
fdefault_policy/Adam_1/update_default_policy/sequential_1/q_out/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power_1*
_output_shapes
: *
dtype0

Udefault_policy/Adam_1/update_default_policy/sequential_1/q_out/bias/ResourceApplyAdamResourceApplyAdam&default_policy/sequential_1/q_out/bias:default_policy/default_policy/sequential_1/q_out/bias/Adam<default_policy/default_policy/sequential_1/q_out/bias/Adam_1ddefault_policy/Adam_1/update_default_policy/sequential_1/q_out/bias/ResourceApplyAdam/ReadVariableOpfdefault_policy/Adam_1/update_default_policy/sequential_1/q_out/bias/ResourceApplyAdam/ReadVariableOp_1#default_policy/Adam_1/learning_ratedefault_policy/Adam_1/beta1default_policy/Adam_1/beta2default_policy/Adam_1/epsilondefault_policy/Identity_11*
T0*9
_class/
-+loc:@default_policy/sequential_1/q_out/bias*&
 _has_manual_control_dependencies(*
use_locking( *
use_nesterov( 
Ѓ
$default_policy/Adam_1/ReadVariableOpReadVariableOpdefault_policy/beta1_power_1[^default_policy/Adam_1/update_default_policy/sequential_1/q_hidden_0/bias/ResourceApplyAdam]^default_policy/Adam_1/update_default_policy/sequential_1/q_hidden_0/kernel/ResourceApplyAdam[^default_policy/Adam_1/update_default_policy/sequential_1/q_hidden_1/bias/ResourceApplyAdam]^default_policy/Adam_1/update_default_policy/sequential_1/q_hidden_1/kernel/ResourceApplyAdamV^default_policy/Adam_1/update_default_policy/sequential_1/q_out/bias/ResourceApplyAdamX^default_policy/Adam_1/update_default_policy/sequential_1/q_out/kernel/ResourceApplyAdam*
_output_shapes
: *
dtype0
Ф
default_policy/Adam_1/mulMul$default_policy/Adam_1/ReadVariableOpdefault_policy/Adam_1/beta1*
T0*>
_class4
20loc:@default_policy/sequential_1/q_hidden_0/bias*
_output_shapes
: 
ў
&default_policy/Adam_1/AssignVariableOpAssignVariableOpdefault_policy/beta1_power_1default_policy/Adam_1/mul*>
_class4
20loc:@default_policy/sequential_1/q_hidden_0/bias*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(

&default_policy/Adam_1/ReadVariableOp_1ReadVariableOpdefault_policy/beta1_power_1'^default_policy/Adam_1/AssignVariableOp[^default_policy/Adam_1/update_default_policy/sequential_1/q_hidden_0/bias/ResourceApplyAdam]^default_policy/Adam_1/update_default_policy/sequential_1/q_hidden_0/kernel/ResourceApplyAdam[^default_policy/Adam_1/update_default_policy/sequential_1/q_hidden_1/bias/ResourceApplyAdam]^default_policy/Adam_1/update_default_policy/sequential_1/q_hidden_1/kernel/ResourceApplyAdamV^default_policy/Adam_1/update_default_policy/sequential_1/q_out/bias/ResourceApplyAdamX^default_policy/Adam_1/update_default_policy/sequential_1/q_out/kernel/ResourceApplyAdam*>
_class4
20loc:@default_policy/sequential_1/q_hidden_0/bias*
_output_shapes
: *
dtype0
Ѕ
&default_policy/Adam_1/ReadVariableOp_2ReadVariableOpdefault_policy/beta2_power_1[^default_policy/Adam_1/update_default_policy/sequential_1/q_hidden_0/bias/ResourceApplyAdam]^default_policy/Adam_1/update_default_policy/sequential_1/q_hidden_0/kernel/ResourceApplyAdam[^default_policy/Adam_1/update_default_policy/sequential_1/q_hidden_1/bias/ResourceApplyAdam]^default_policy/Adam_1/update_default_policy/sequential_1/q_hidden_1/kernel/ResourceApplyAdamV^default_policy/Adam_1/update_default_policy/sequential_1/q_out/bias/ResourceApplyAdamX^default_policy/Adam_1/update_default_policy/sequential_1/q_out/kernel/ResourceApplyAdam*
_output_shapes
: *
dtype0
Ш
default_policy/Adam_1/mul_1Mul&default_policy/Adam_1/ReadVariableOp_2default_policy/Adam_1/beta2*
T0*>
_class4
20loc:@default_policy/sequential_1/q_hidden_0/bias*
_output_shapes
: 

(default_policy/Adam_1/AssignVariableOp_1AssignVariableOpdefault_policy/beta2_power_1default_policy/Adam_1/mul_1*>
_class4
20loc:@default_policy/sequential_1/q_hidden_0/bias*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(

&default_policy/Adam_1/ReadVariableOp_3ReadVariableOpdefault_policy/beta2_power_1)^default_policy/Adam_1/AssignVariableOp_1[^default_policy/Adam_1/update_default_policy/sequential_1/q_hidden_0/bias/ResourceApplyAdam]^default_policy/Adam_1/update_default_policy/sequential_1/q_hidden_0/kernel/ResourceApplyAdam[^default_policy/Adam_1/update_default_policy/sequential_1/q_hidden_1/bias/ResourceApplyAdam]^default_policy/Adam_1/update_default_policy/sequential_1/q_hidden_1/kernel/ResourceApplyAdamV^default_policy/Adam_1/update_default_policy/sequential_1/q_out/bias/ResourceApplyAdamX^default_policy/Adam_1/update_default_policy/sequential_1/q_out/kernel/ResourceApplyAdam*>
_class4
20loc:@default_policy/sequential_1/q_hidden_0/bias*
_output_shapes
: *
dtype0
У
default_policy/Adam_1NoOp'^default_policy/Adam_1/AssignVariableOp)^default_policy/Adam_1/AssignVariableOp_1[^default_policy/Adam_1/update_default_policy/sequential_1/q_hidden_0/bias/ResourceApplyAdam]^default_policy/Adam_1/update_default_policy/sequential_1/q_hidden_0/kernel/ResourceApplyAdam[^default_policy/Adam_1/update_default_policy/sequential_1/q_hidden_1/bias/ResourceApplyAdam]^default_policy/Adam_1/update_default_policy/sequential_1/q_hidden_1/kernel/ResourceApplyAdamV^default_policy/Adam_1/update_default_policy/sequential_1/q_out/bias/ResourceApplyAdamX^default_policy/Adam_1/update_default_policy/sequential_1/q_out/kernel/ResourceApplyAdam*&
 _has_manual_control_dependencies(
Р
6default_policy/beta1_power_2/Initializer/initial_valueConst*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes
: *
dtype0*
valueB
 *fff?
љ
default_policy/beta1_power_2VarHandleOp*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *-
shared_namedefault_policy/beta1_power_2
Ю
=default_policy/beta1_power_2/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/beta1_power_2*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes
: 
и
#default_policy/beta1_power_2/AssignAssignVariableOpdefault_policy/beta1_power_26default_policy/beta1_power_2/Initializer/initial_value*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Ъ
0default_policy/beta1_power_2/Read/ReadVariableOpReadVariableOpdefault_policy/beta1_power_2*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes
: *
dtype0
Р
6default_policy/beta2_power_2/Initializer/initial_valueConst*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes
: *
dtype0*
valueB
 *wО?
љ
default_policy/beta2_power_2VarHandleOp*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *-
shared_namedefault_policy/beta2_power_2
Ю
=default_policy/beta2_power_2/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/beta2_power_2*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes
: 
и
#default_policy/beta2_power_2/AssignAssignVariableOpdefault_policy/beta2_power_26default_policy/beta2_power_2/Initializer/initial_value*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Ъ
0default_policy/beta2_power_2/Read/ReadVariableOpReadVariableOpdefault_policy/beta2_power_2*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes
: *
dtype0

hdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam/Initializer/zeros/shape_as_tensorConst*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_0/kernel*
_output_shapes
:*
dtype0*
valueB"     
ъ
^default_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam/Initializer/zeros/ConstConst*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
Ў
Xdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam/Initializer/zerosFillhdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam/Initializer/zeros/shape_as_tensor^default_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam/Initializer/zeros/Const*
T0*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_0/kernel* 
_output_shapes
:
*

index_type0
й
Fdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/AdamVarHandleOp*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_0/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:
*W
shared_nameHFdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam
Є
gdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpFdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_0/kernel*
_output_shapes
: 
Ю
Mdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam/AssignAssignVariableOpFdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/AdamXdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Њ
Zdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam/Read/ReadVariableOpReadVariableOpFdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_0/kernel* 
_output_shapes
:
*
dtype0

jdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_0/kernel*
_output_shapes
:*
dtype0*
valueB"     
ь
`default_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam_1/Initializer/zeros/ConstConst*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_0/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
Д
Zdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam_1/Initializer/zerosFilljdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam_1/Initializer/zeros/shape_as_tensor`default_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam_1/Initializer/zeros/Const*
T0*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_0/kernel* 
_output_shapes
:
*

index_type0
н
Hdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam_1VarHandleOp*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_0/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:
*Y
shared_nameJHdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam_1
Ј
idefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpHdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam_1*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_0/kernel*
_output_shapes
: 
д
Odefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam_1/AssignAssignVariableOpHdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam_1Zdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Ў
\default_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam_1/Read/ReadVariableOpReadVariableOpHdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam_1*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_0/kernel* 
_output_shapes
:
*
dtype0
ъ
Vdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam/Initializer/zerosConst*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes	
:*
dtype0*
valueB*    
Ю
Ddefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/AdamVarHandleOp*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*U
shared_nameFDdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam

edefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpDdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes
: 
Ш
Kdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam/AssignAssignVariableOpDdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/AdamVdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Xdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam/Read/ReadVariableOpReadVariableOpDdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes	
:*
dtype0
ь
Xdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam_1/Initializer/zerosConst*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes	
:*
dtype0*
valueB*    
в
Fdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam_1VarHandleOp*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*W
shared_nameHFdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam_1
Ђ
gdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpFdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam_1*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes
: 
Ю
Mdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam_1/AssignAssignVariableOpFdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam_1Xdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Ѓ
Zdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam_1/Read/ReadVariableOpReadVariableOpFdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam_1*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes	
:*
dtype0

hdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_1/kernel*
_output_shapes
:*
dtype0*
valueB"      
ъ
^default_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam/Initializer/zeros/ConstConst*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
Ў
Xdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam/Initializer/zerosFillhdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam/Initializer/zeros/shape_as_tensor^default_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam/Initializer/zeros/Const*
T0*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_1/kernel* 
_output_shapes
:
*

index_type0
й
Fdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/AdamVarHandleOp*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_1/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:
*W
shared_nameHFdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam
Є
gdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpFdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_1/kernel*
_output_shapes
: 
Ю
Mdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam/AssignAssignVariableOpFdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/AdamXdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Њ
Zdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam/Read/ReadVariableOpReadVariableOpFdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_1/kernel* 
_output_shapes
:
*
dtype0

jdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_1/kernel*
_output_shapes
:*
dtype0*
valueB"      
ь
`default_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam_1/Initializer/zeros/ConstConst*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
Д
Zdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam_1/Initializer/zerosFilljdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor`default_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam_1/Initializer/zeros/Const*
T0*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_1/kernel* 
_output_shapes
:
*

index_type0
н
Hdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam_1VarHandleOp*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_1/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:
*Y
shared_nameJHdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam_1
Ј
idefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpHdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam_1*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_1/kernel*
_output_shapes
: 
д
Odefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam_1/AssignAssignVariableOpHdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam_1Zdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Ў
\default_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam_1/Read/ReadVariableOpReadVariableOpHdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam_1*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_1/kernel* 
_output_shapes
:
*
dtype0
ъ
Vdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam/Initializer/zerosConst*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_1/bias*
_output_shapes	
:*
dtype0*
valueB*    
Ю
Ddefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/AdamVarHandleOp*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_1/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*U
shared_nameFDdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam

edefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpDdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_1/bias*
_output_shapes
: 
Ш
Kdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam/AssignAssignVariableOpDdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/AdamVdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Xdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam/Read/ReadVariableOpReadVariableOpDdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_1/bias*
_output_shapes	
:*
dtype0
ь
Xdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam_1/Initializer/zerosConst*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_1/bias*
_output_shapes	
:*
dtype0*
valueB*    
в
Fdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam_1VarHandleOp*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_1/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*W
shared_nameHFdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam_1
Ђ
gdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpFdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam_1*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_1/bias*
_output_shapes
: 
Ю
Mdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam_1/AssignAssignVariableOpFdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam_1Xdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Ѓ
Zdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam_1/Read/ReadVariableOpReadVariableOpFdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam_1*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_1/bias*
_output_shapes	
:*
dtype0
ь
Sdefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam/Initializer/zerosConst*@
_class6
42loc:@default_policy/sequential_2/twin_q_out/kernel*
_output_shapes
:	*
dtype0*
valueB	*    
Щ
Adefault_policy/default_policy/sequential_2/twin_q_out/kernel/AdamVarHandleOp*@
_class6
42loc:@default_policy/sequential_2/twin_q_out/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:	*R
shared_nameCAdefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam

bdefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam*@
_class6
42loc:@default_policy/sequential_2/twin_q_out/kernel*
_output_shapes
: 
П
Hdefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam/AssignAssignVariableOpAdefault_policy/default_policy/sequential_2/twin_q_out/kernel/AdamSdefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Udefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam/Read/ReadVariableOpReadVariableOpAdefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam*@
_class6
42loc:@default_policy/sequential_2/twin_q_out/kernel*
_output_shapes
:	*
dtype0
ю
Udefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam_1/Initializer/zerosConst*@
_class6
42loc:@default_policy/sequential_2/twin_q_out/kernel*
_output_shapes
:	*
dtype0*
valueB	*    
Э
Cdefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam_1VarHandleOp*@
_class6
42loc:@default_policy/sequential_2/twin_q_out/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:	*T
shared_nameECdefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam_1

ddefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpCdefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam_1*@
_class6
42loc:@default_policy/sequential_2/twin_q_out/kernel*
_output_shapes
: 
Х
Jdefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam_1/AssignAssignVariableOpCdefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam_1Udefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Wdefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam_1/Read/ReadVariableOpReadVariableOpCdefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam_1*@
_class6
42loc:@default_policy/sequential_2/twin_q_out/kernel*
_output_shapes
:	*
dtype0
о
Qdefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam/Initializer/zerosConst*>
_class4
20loc:@default_policy/sequential_2/twin_q_out/bias*
_output_shapes
:*
dtype0*
valueB*    
О
?default_policy/default_policy/sequential_2/twin_q_out/bias/AdamVarHandleOp*>
_class4
20loc:@default_policy/sequential_2/twin_q_out/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*P
shared_nameA?default_policy/default_policy/sequential_2/twin_q_out/bias/Adam

`default_policy/default_policy/sequential_2/twin_q_out/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp?default_policy/default_policy/sequential_2/twin_q_out/bias/Adam*>
_class4
20loc:@default_policy/sequential_2/twin_q_out/bias*
_output_shapes
: 
Й
Fdefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam/AssignAssignVariableOp?default_policy/default_policy/sequential_2/twin_q_out/bias/AdamQdefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Sdefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam/Read/ReadVariableOpReadVariableOp?default_policy/default_policy/sequential_2/twin_q_out/bias/Adam*>
_class4
20loc:@default_policy/sequential_2/twin_q_out/bias*
_output_shapes
:*
dtype0
р
Sdefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam_1/Initializer/zerosConst*>
_class4
20loc:@default_policy/sequential_2/twin_q_out/bias*
_output_shapes
:*
dtype0*
valueB*    
Т
Adefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam_1VarHandleOp*>
_class4
20loc:@default_policy/sequential_2/twin_q_out/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*R
shared_nameCAdefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam_1

bdefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam_1*>
_class4
20loc:@default_policy/sequential_2/twin_q_out/bias*
_output_shapes
: 
П
Hdefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam_1/AssignAssignVariableOpAdefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam_1Sdefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

Udefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam_1/Read/ReadVariableOpReadVariableOpAdefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam_1*>
_class4
20loc:@default_policy/sequential_2/twin_q_out/bias*
_output_shapes
:*
dtype0
h
#default_policy/Adam_2/learning_rateConst*
_output_shapes
: *
dtype0*
valueB
 *Зб8
`
default_policy/Adam_2/beta1Const*
_output_shapes
: *
dtype0*
valueB
 *fff?
`
default_policy/Adam_2/beta2Const*
_output_shapes
: *
dtype0*
valueB
 *wО?
b
default_policy/Adam_2/epsilonConst*
_output_shapes
: *
dtype0*
valueB
 *wЬ+2
Х
pdefault_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_0/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power_2*
_output_shapes
: *
dtype0
Ч
rdefault_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_0/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power_2*
_output_shapes
: *
dtype0
л
adefault_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_0/kernel/ResourceApplyAdamResourceApplyAdam2default_policy/sequential_2/twin_q_hidden_0/kernelFdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/AdamHdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam_1pdefault_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_0/kernel/ResourceApplyAdam/ReadVariableOprdefault_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_0/kernel/ResourceApplyAdam/ReadVariableOp_1#default_policy/Adam_2/learning_ratedefault_policy/Adam_2/beta1default_policy/Adam_2/beta2default_policy/Adam_2/epsilondefault_policy/Identity_12*
T0*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_0/kernel*&
 _has_manual_control_dependencies(*
use_locking( *
use_nesterov( 
У
ndefault_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_0/bias/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power_2*
_output_shapes
: *
dtype0
Х
pdefault_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_0/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power_2*
_output_shapes
: *
dtype0
Э
_default_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_0/bias/ResourceApplyAdamResourceApplyAdam0default_policy/sequential_2/twin_q_hidden_0/biasDdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/AdamFdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam_1ndefault_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_0/bias/ResourceApplyAdam/ReadVariableOppdefault_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_0/bias/ResourceApplyAdam/ReadVariableOp_1#default_policy/Adam_2/learning_ratedefault_policy/Adam_2/beta1default_policy/Adam_2/beta2default_policy/Adam_2/epsilondefault_policy/Identity_13*
T0*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_0/bias*&
 _has_manual_control_dependencies(*
use_locking( *
use_nesterov( 
Х
pdefault_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_1/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power_2*
_output_shapes
: *
dtype0
Ч
rdefault_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_1/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power_2*
_output_shapes
: *
dtype0
л
adefault_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_1/kernel/ResourceApplyAdamResourceApplyAdam2default_policy/sequential_2/twin_q_hidden_1/kernelFdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/AdamHdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam_1pdefault_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_1/kernel/ResourceApplyAdam/ReadVariableOprdefault_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_1/kernel/ResourceApplyAdam/ReadVariableOp_1#default_policy/Adam_2/learning_ratedefault_policy/Adam_2/beta1default_policy/Adam_2/beta2default_policy/Adam_2/epsilondefault_policy/Identity_14*
T0*E
_class;
97loc:@default_policy/sequential_2/twin_q_hidden_1/kernel*&
 _has_manual_control_dependencies(*
use_locking( *
use_nesterov( 
У
ndefault_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_1/bias/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power_2*
_output_shapes
: *
dtype0
Х
pdefault_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_1/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power_2*
_output_shapes
: *
dtype0
Э
_default_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_1/bias/ResourceApplyAdamResourceApplyAdam0default_policy/sequential_2/twin_q_hidden_1/biasDdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/AdamFdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam_1ndefault_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_1/bias/ResourceApplyAdam/ReadVariableOppdefault_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_1/bias/ResourceApplyAdam/ReadVariableOp_1#default_policy/Adam_2/learning_ratedefault_policy/Adam_2/beta1default_policy/Adam_2/beta2default_policy/Adam_2/epsilondefault_policy/Identity_15*
T0*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_1/bias*&
 _has_manual_control_dependencies(*
use_locking( *
use_nesterov( 
Р
kdefault_policy/Adam_2/update_default_policy/sequential_2/twin_q_out/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power_2*
_output_shapes
: *
dtype0
Т
mdefault_policy/Adam_2/update_default_policy/sequential_2/twin_q_out/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power_2*
_output_shapes
: *
dtype0
И
\default_policy/Adam_2/update_default_policy/sequential_2/twin_q_out/kernel/ResourceApplyAdamResourceApplyAdam-default_policy/sequential_2/twin_q_out/kernelAdefault_policy/default_policy/sequential_2/twin_q_out/kernel/AdamCdefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam_1kdefault_policy/Adam_2/update_default_policy/sequential_2/twin_q_out/kernel/ResourceApplyAdam/ReadVariableOpmdefault_policy/Adam_2/update_default_policy/sequential_2/twin_q_out/kernel/ResourceApplyAdam/ReadVariableOp_1#default_policy/Adam_2/learning_ratedefault_policy/Adam_2/beta1default_policy/Adam_2/beta2default_policy/Adam_2/epsilondefault_policy/Identity_16*
T0*@
_class6
42loc:@default_policy/sequential_2/twin_q_out/kernel*&
 _has_manual_control_dependencies(*
use_locking( *
use_nesterov( 
О
idefault_policy/Adam_2/update_default_policy/sequential_2/twin_q_out/bias/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power_2*
_output_shapes
: *
dtype0
Р
kdefault_policy/Adam_2/update_default_policy/sequential_2/twin_q_out/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power_2*
_output_shapes
: *
dtype0
Њ
Zdefault_policy/Adam_2/update_default_policy/sequential_2/twin_q_out/bias/ResourceApplyAdamResourceApplyAdam+default_policy/sequential_2/twin_q_out/bias?default_policy/default_policy/sequential_2/twin_q_out/bias/AdamAdefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam_1idefault_policy/Adam_2/update_default_policy/sequential_2/twin_q_out/bias/ResourceApplyAdam/ReadVariableOpkdefault_policy/Adam_2/update_default_policy/sequential_2/twin_q_out/bias/ResourceApplyAdam/ReadVariableOp_1#default_policy/Adam_2/learning_ratedefault_policy/Adam_2/beta1default_policy/Adam_2/beta2default_policy/Adam_2/epsilondefault_policy/Identity_17*
T0*>
_class4
20loc:@default_policy/sequential_2/twin_q_out/bias*&
 _has_manual_control_dependencies(*
use_locking( *
use_nesterov( 
С
$default_policy/Adam_2/ReadVariableOpReadVariableOpdefault_policy/beta1_power_2`^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_0/bias/ResourceApplyAdamb^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_0/kernel/ResourceApplyAdam`^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_1/bias/ResourceApplyAdamb^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_1/kernel/ResourceApplyAdam[^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_out/bias/ResourceApplyAdam]^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_out/kernel/ResourceApplyAdam*
_output_shapes
: *
dtype0
Щ
default_policy/Adam_2/mulMul$default_policy/Adam_2/ReadVariableOpdefault_policy/Adam_2/beta1*
T0*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes
: 

&default_policy/Adam_2/AssignVariableOpAssignVariableOpdefault_policy/beta1_power_2default_policy/Adam_2/mul*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_0/bias*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Б
&default_policy/Adam_2/ReadVariableOp_1ReadVariableOpdefault_policy/beta1_power_2'^default_policy/Adam_2/AssignVariableOp`^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_0/bias/ResourceApplyAdamb^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_0/kernel/ResourceApplyAdam`^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_1/bias/ResourceApplyAdamb^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_1/kernel/ResourceApplyAdam[^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_out/bias/ResourceApplyAdam]^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_out/kernel/ResourceApplyAdam*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes
: *
dtype0
У
&default_policy/Adam_2/ReadVariableOp_2ReadVariableOpdefault_policy/beta2_power_2`^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_0/bias/ResourceApplyAdamb^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_0/kernel/ResourceApplyAdam`^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_1/bias/ResourceApplyAdamb^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_1/kernel/ResourceApplyAdam[^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_out/bias/ResourceApplyAdam]^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_out/kernel/ResourceApplyAdam*
_output_shapes
: *
dtype0
Э
default_policy/Adam_2/mul_1Mul&default_policy/Adam_2/ReadVariableOp_2default_policy/Adam_2/beta2*
T0*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes
: 

(default_policy/Adam_2/AssignVariableOp_1AssignVariableOpdefault_policy/beta2_power_2default_policy/Adam_2/mul_1*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_0/bias*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Г
&default_policy/Adam_2/ReadVariableOp_3ReadVariableOpdefault_policy/beta2_power_2)^default_policy/Adam_2/AssignVariableOp_1`^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_0/bias/ResourceApplyAdamb^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_0/kernel/ResourceApplyAdam`^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_1/bias/ResourceApplyAdamb^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_1/kernel/ResourceApplyAdam[^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_out/bias/ResourceApplyAdam]^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_out/kernel/ResourceApplyAdam*C
_class9
75loc:@default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes
: *
dtype0
с
default_policy/Adam_2NoOp'^default_policy/Adam_2/AssignVariableOp)^default_policy/Adam_2/AssignVariableOp_1`^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_0/bias/ResourceApplyAdamb^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_0/kernel/ResourceApplyAdam`^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_1/bias/ResourceApplyAdamb^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_hidden_1/kernel/ResourceApplyAdam[^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_out/bias/ResourceApplyAdam]^default_policy/Adam_2/update_default_policy/sequential_2/twin_q_out/kernel/ResourceApplyAdam*&
 _has_manual_control_dependencies(
Ј
6default_policy/beta1_power_3/Initializer/initial_valueConst*+
_class!
loc:@default_policy/log_alpha*
_output_shapes
: *
dtype0*
valueB
 *fff?
с
default_policy/beta1_power_3VarHandleOp*+
_class!
loc:@default_policy/log_alpha*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *-
shared_namedefault_policy/beta1_power_3
Ж
=default_policy/beta1_power_3/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/beta1_power_3*+
_class!
loc:@default_policy/log_alpha*
_output_shapes
: 
и
#default_policy/beta1_power_3/AssignAssignVariableOpdefault_policy/beta1_power_36default_policy/beta1_power_3/Initializer/initial_value*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
В
0default_policy/beta1_power_3/Read/ReadVariableOpReadVariableOpdefault_policy/beta1_power_3*+
_class!
loc:@default_policy/log_alpha*
_output_shapes
: *
dtype0
Ј
6default_policy/beta2_power_3/Initializer/initial_valueConst*+
_class!
loc:@default_policy/log_alpha*
_output_shapes
: *
dtype0*
valueB
 *wО?
с
default_policy/beta2_power_3VarHandleOp*+
_class!
loc:@default_policy/log_alpha*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *-
shared_namedefault_policy/beta2_power_3
Ж
=default_policy/beta2_power_3/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/beta2_power_3*+
_class!
loc:@default_policy/log_alpha*
_output_shapes
: 
и
#default_policy/beta2_power_3/AssignAssignVariableOpdefault_policy/beta2_power_36default_policy/beta2_power_3/Initializer/initial_value*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
В
0default_policy/beta2_power_3/Read/ReadVariableOpReadVariableOpdefault_policy/beta2_power_3*+
_class!
loc:@default_policy/log_alpha*
_output_shapes
: *
dtype0
А
>default_policy/default_policy/log_alpha/Adam/Initializer/zerosConst*+
_class!
loc:@default_policy/log_alpha*
_output_shapes
: *
dtype0*
valueB
 *    

,default_policy/default_policy/log_alpha/AdamVarHandleOp*+
_class!
loc:@default_policy/log_alpha*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *=
shared_name.,default_policy/default_policy/log_alpha/Adam
ж
Mdefault_policy/default_policy/log_alpha/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp,default_policy/default_policy/log_alpha/Adam*+
_class!
loc:@default_policy/log_alpha*
_output_shapes
: 

3default_policy/default_policy/log_alpha/Adam/AssignAssignVariableOp,default_policy/default_policy/log_alpha/Adam>default_policy/default_policy/log_alpha/Adam/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
в
@default_policy/default_policy/log_alpha/Adam/Read/ReadVariableOpReadVariableOp,default_policy/default_policy/log_alpha/Adam*+
_class!
loc:@default_policy/log_alpha*
_output_shapes
: *
dtype0
В
@default_policy/default_policy/log_alpha/Adam_1/Initializer/zerosConst*+
_class!
loc:@default_policy/log_alpha*
_output_shapes
: *
dtype0*
valueB
 *    

.default_policy/default_policy/log_alpha/Adam_1VarHandleOp*+
_class!
loc:@default_policy/log_alpha*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *?
shared_name0.default_policy/default_policy/log_alpha/Adam_1
к
Odefault_policy/default_policy/log_alpha/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp.default_policy/default_policy/log_alpha/Adam_1*+
_class!
loc:@default_policy/log_alpha*
_output_shapes
: 

5default_policy/default_policy/log_alpha/Adam_1/AssignAssignVariableOp.default_policy/default_policy/log_alpha/Adam_1@default_policy/default_policy/log_alpha/Adam_1/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
ж
Bdefault_policy/default_policy/log_alpha/Adam_1/Read/ReadVariableOpReadVariableOp.default_policy/default_policy/log_alpha/Adam_1*+
_class!
loc:@default_policy/log_alpha*
_output_shapes
: *
dtype0
h
#default_policy/Adam_3/learning_rateConst*
_output_shapes
: *
dtype0*
valueB
 *Зб8
`
default_policy/Adam_3/beta1Const*
_output_shapes
: *
dtype0*
valueB
 *fff?
`
default_policy/Adam_3/beta2Const*
_output_shapes
: *
dtype0*
valueB
 *wО?
b
default_policy/Adam_3/epsilonConst*
_output_shapes
: *
dtype0*
valueB
 *wЬ+2
Ћ
Vdefault_policy/Adam_3/update_default_policy/log_alpha/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power_3*
_output_shapes
: *
dtype0
­
Xdefault_policy/Adam_3/update_default_policy/log_alpha/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power_3*
_output_shapes
: *
dtype0
Ѕ
Gdefault_policy/Adam_3/update_default_policy/log_alpha/ResourceApplyAdamResourceApplyAdamdefault_policy/log_alpha,default_policy/default_policy/log_alpha/Adam.default_policy/default_policy/log_alpha/Adam_1Vdefault_policy/Adam_3/update_default_policy/log_alpha/ResourceApplyAdam/ReadVariableOpXdefault_policy/Adam_3/update_default_policy/log_alpha/ResourceApplyAdam/ReadVariableOp_1#default_policy/Adam_3/learning_ratedefault_policy/Adam_3/beta1default_policy/Adam_3/beta2default_policy/Adam_3/epsilondefault_policy/Identity_18*
T0*+
_class!
loc:@default_policy/log_alpha*&
 _has_manual_control_dependencies(*
use_locking( *
use_nesterov( 
У
$default_policy/Adam_3/ReadVariableOpReadVariableOpdefault_policy/beta1_power_3H^default_policy/Adam_3/update_default_policy/log_alpha/ResourceApplyAdam*
_output_shapes
: *
dtype0
Б
default_policy/Adam_3/mulMul$default_policy/Adam_3/ReadVariableOpdefault_policy/Adam_3/beta1*
T0*+
_class!
loc:@default_policy/log_alpha*
_output_shapes
: 
ы
&default_policy/Adam_3/AssignVariableOpAssignVariableOpdefault_policy/beta1_power_3default_policy/Adam_3/mul*+
_class!
loc:@default_policy/log_alpha*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(

&default_policy/Adam_3/ReadVariableOp_1ReadVariableOpdefault_policy/beta1_power_3'^default_policy/Adam_3/AssignVariableOpH^default_policy/Adam_3/update_default_policy/log_alpha/ResourceApplyAdam*+
_class!
loc:@default_policy/log_alpha*
_output_shapes
: *
dtype0
Х
&default_policy/Adam_3/ReadVariableOp_2ReadVariableOpdefault_policy/beta2_power_3H^default_policy/Adam_3/update_default_policy/log_alpha/ResourceApplyAdam*
_output_shapes
: *
dtype0
Е
default_policy/Adam_3/mul_1Mul&default_policy/Adam_3/ReadVariableOp_2default_policy/Adam_3/beta2*
T0*+
_class!
loc:@default_policy/log_alpha*
_output_shapes
: 
я
(default_policy/Adam_3/AssignVariableOp_1AssignVariableOpdefault_policy/beta2_power_3default_policy/Adam_3/mul_1*+
_class!
loc:@default_policy/log_alpha*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(

&default_policy/Adam_3/ReadVariableOp_3ReadVariableOpdefault_policy/beta2_power_3)^default_policy/Adam_3/AssignVariableOp_1H^default_policy/Adam_3/update_default_policy/log_alpha/ResourceApplyAdam*+
_class!
loc:@default_policy/log_alpha*
_output_shapes
: *
dtype0
ъ
default_policy/Adam_3/updateNoOp'^default_policy/Adam_3/AssignVariableOp)^default_policy/Adam_3/AssignVariableOp_1H^default_policy/Adam_3/update_default_policy/log_alpha/ResourceApplyAdam*&
 _has_manual_control_dependencies(
Ћ
default_policy/Adam_3/valueConst^default_policy/Adam_3/update*-
_class#
!loc:@default_policy/global_step*
_output_shapes
: *
dtype0	*
value	B	 R
ц
default_policy/Adam_3	AssignAdddefault_policy/global_stepdefault_policy/Adam_3/value*
T0	*-
_class#
!loc:@default_policy/global_step*&
 _has_manual_control_dependencies(*
_output_shapes
: *
use_locking( 

default_policy/group_depsNoOp^default_policy/Adam^default_policy/Adam_1^default_policy/Adam_2^default_policy/Adam_3
Э)
default_policy/init_1NoOp"^default_policy/beta1_power/Assign$^default_policy/beta1_power_1/Assign$^default_policy/beta1_power_2/Assign$^default_policy/beta1_power_3/Assign"^default_policy/beta2_power/Assign$^default_policy/beta2_power_1/Assign$^default_policy/beta2_power_2/Assign$^default_policy/beta2_power_3/Assign4^default_policy/default_policy/log_alpha/Adam/Assign6^default_policy/default_policy/log_alpha/Adam_1/AssignC^default_policy/default_policy/sequential/action_1/bias/Adam/AssignE^default_policy/default_policy/sequential/action_1/bias/Adam_1/AssignE^default_policy/default_policy/sequential/action_1/kernel/Adam/AssignG^default_policy/default_policy/sequential/action_1/kernel/Adam_1/AssignC^default_policy/default_policy/sequential/action_2/bias/Adam/AssignE^default_policy/default_policy/sequential/action_2/bias/Adam_1/AssignE^default_policy/default_policy/sequential/action_2/kernel/Adam/AssignG^default_policy/default_policy/sequential/action_2/kernel/Adam_1/AssignE^default_policy/default_policy/sequential/action_out/bias/Adam/AssignG^default_policy/default_policy/sequential/action_out/bias/Adam_1/AssignG^default_policy/default_policy/sequential/action_out/kernel/Adam/AssignI^default_policy/default_policy/sequential/action_out/kernel/Adam_1/AssignG^default_policy/default_policy/sequential_1/q_hidden_0/bias/Adam/AssignI^default_policy/default_policy/sequential_1/q_hidden_0/bias/Adam_1/AssignI^default_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam/AssignK^default_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam_1/AssignG^default_policy/default_policy/sequential_1/q_hidden_1/bias/Adam/AssignI^default_policy/default_policy/sequential_1/q_hidden_1/bias/Adam_1/AssignI^default_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam/AssignK^default_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam_1/AssignB^default_policy/default_policy/sequential_1/q_out/bias/Adam/AssignD^default_policy/default_policy/sequential_1/q_out/bias/Adam_1/AssignD^default_policy/default_policy/sequential_1/q_out/kernel/Adam/AssignF^default_policy/default_policy/sequential_1/q_out/kernel/Adam_1/AssignL^default_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam/AssignN^default_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam_1/AssignN^default_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam/AssignP^default_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam_1/AssignL^default_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam/AssignN^default_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam_1/AssignN^default_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam/AssignP^default_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam_1/AssignG^default_policy/default_policy/sequential_2/twin_q_out/bias/Adam/AssignI^default_policy/default_policy/sequential_2/twin_q_out/bias/Adam_1/AssignI^default_policy/default_policy/sequential_2/twin_q_out/kernel/Adam/AssignK^default_policy/default_policy/sequential_2/twin_q_out/kernel/Adam_1/Assign"^default_policy/global_step/Assign ^default_policy/log_alpha/Assign"^default_policy/log_alpha_1/Assign/^default_policy/sequential/action_1/bias/Assign1^default_policy/sequential/action_1/kernel/Assign/^default_policy/sequential/action_2/bias/Assign1^default_policy/sequential/action_2/kernel/Assign1^default_policy/sequential/action_out/bias/Assign3^default_policy/sequential/action_out/kernel/Assign3^default_policy/sequential_1/q_hidden_0/bias/Assign5^default_policy/sequential_1/q_hidden_0/kernel/Assign3^default_policy/sequential_1/q_hidden_1/bias/Assign5^default_policy/sequential_1/q_hidden_1/kernel/Assign.^default_policy/sequential_1/q_out/bias/Assign0^default_policy/sequential_1/q_out/kernel/Assign8^default_policy/sequential_2/twin_q_hidden_0/bias/Assign:^default_policy/sequential_2/twin_q_hidden_0/kernel/Assign8^default_policy/sequential_2/twin_q_hidden_1/bias/Assign:^default_policy/sequential_2/twin_q_hidden_1/kernel/Assign3^default_policy/sequential_2/twin_q_out/bias/Assign5^default_policy/sequential_2/twin_q_out/kernel/Assign1^default_policy/sequential_3/action_1/bias/Assign3^default_policy/sequential_3/action_1/kernel/Assign1^default_policy/sequential_3/action_2/bias/Assign3^default_policy/sequential_3/action_2/kernel/Assign3^default_policy/sequential_3/action_out/bias/Assign5^default_policy/sequential_3/action_out/kernel/Assign3^default_policy/sequential_4/q_hidden_0/bias/Assign5^default_policy/sequential_4/q_hidden_0/kernel/Assign3^default_policy/sequential_4/q_hidden_1/bias/Assign5^default_policy/sequential_4/q_hidden_1/kernel/Assign.^default_policy/sequential_4/q_out/bias/Assign0^default_policy/sequential_4/q_out/kernel/Assign8^default_policy/sequential_5/twin_q_hidden_0/bias/Assign:^default_policy/sequential_5/twin_q_hidden_0/kernel/Assign8^default_policy/sequential_5/twin_q_hidden_1/bias/Assign:^default_policy/sequential_5/twin_q_hidden_1/kernel/Assign3^default_policy/sequential_5/twin_q_out/bias/Assign5^default_policy/sequential_5/twin_q_out/kernel/Assign%^default_policy/value_out/bias/Assign'^default_policy/value_out/kernel/Assign'^default_policy/value_out_1/bias/Assign)^default_policy/value_out_1/kernel/Assign
Э)
default_policy/init_2NoOp"^default_policy/beta1_power/Assign$^default_policy/beta1_power_1/Assign$^default_policy/beta1_power_2/Assign$^default_policy/beta1_power_3/Assign"^default_policy/beta2_power/Assign$^default_policy/beta2_power_1/Assign$^default_policy/beta2_power_2/Assign$^default_policy/beta2_power_3/Assign4^default_policy/default_policy/log_alpha/Adam/Assign6^default_policy/default_policy/log_alpha/Adam_1/AssignC^default_policy/default_policy/sequential/action_1/bias/Adam/AssignE^default_policy/default_policy/sequential/action_1/bias/Adam_1/AssignE^default_policy/default_policy/sequential/action_1/kernel/Adam/AssignG^default_policy/default_policy/sequential/action_1/kernel/Adam_1/AssignC^default_policy/default_policy/sequential/action_2/bias/Adam/AssignE^default_policy/default_policy/sequential/action_2/bias/Adam_1/AssignE^default_policy/default_policy/sequential/action_2/kernel/Adam/AssignG^default_policy/default_policy/sequential/action_2/kernel/Adam_1/AssignE^default_policy/default_policy/sequential/action_out/bias/Adam/AssignG^default_policy/default_policy/sequential/action_out/bias/Adam_1/AssignG^default_policy/default_policy/sequential/action_out/kernel/Adam/AssignI^default_policy/default_policy/sequential/action_out/kernel/Adam_1/AssignG^default_policy/default_policy/sequential_1/q_hidden_0/bias/Adam/AssignI^default_policy/default_policy/sequential_1/q_hidden_0/bias/Adam_1/AssignI^default_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam/AssignK^default_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam_1/AssignG^default_policy/default_policy/sequential_1/q_hidden_1/bias/Adam/AssignI^default_policy/default_policy/sequential_1/q_hidden_1/bias/Adam_1/AssignI^default_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam/AssignK^default_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam_1/AssignB^default_policy/default_policy/sequential_1/q_out/bias/Adam/AssignD^default_policy/default_policy/sequential_1/q_out/bias/Adam_1/AssignD^default_policy/default_policy/sequential_1/q_out/kernel/Adam/AssignF^default_policy/default_policy/sequential_1/q_out/kernel/Adam_1/AssignL^default_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam/AssignN^default_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam_1/AssignN^default_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam/AssignP^default_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam_1/AssignL^default_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam/AssignN^default_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam_1/AssignN^default_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam/AssignP^default_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam_1/AssignG^default_policy/default_policy/sequential_2/twin_q_out/bias/Adam/AssignI^default_policy/default_policy/sequential_2/twin_q_out/bias/Adam_1/AssignI^default_policy/default_policy/sequential_2/twin_q_out/kernel/Adam/AssignK^default_policy/default_policy/sequential_2/twin_q_out/kernel/Adam_1/Assign"^default_policy/global_step/Assign ^default_policy/log_alpha/Assign"^default_policy/log_alpha_1/Assign/^default_policy/sequential/action_1/bias/Assign1^default_policy/sequential/action_1/kernel/Assign/^default_policy/sequential/action_2/bias/Assign1^default_policy/sequential/action_2/kernel/Assign1^default_policy/sequential/action_out/bias/Assign3^default_policy/sequential/action_out/kernel/Assign3^default_policy/sequential_1/q_hidden_0/bias/Assign5^default_policy/sequential_1/q_hidden_0/kernel/Assign3^default_policy/sequential_1/q_hidden_1/bias/Assign5^default_policy/sequential_1/q_hidden_1/kernel/Assign.^default_policy/sequential_1/q_out/bias/Assign0^default_policy/sequential_1/q_out/kernel/Assign8^default_policy/sequential_2/twin_q_hidden_0/bias/Assign:^default_policy/sequential_2/twin_q_hidden_0/kernel/Assign8^default_policy/sequential_2/twin_q_hidden_1/bias/Assign:^default_policy/sequential_2/twin_q_hidden_1/kernel/Assign3^default_policy/sequential_2/twin_q_out/bias/Assign5^default_policy/sequential_2/twin_q_out/kernel/Assign1^default_policy/sequential_3/action_1/bias/Assign3^default_policy/sequential_3/action_1/kernel/Assign1^default_policy/sequential_3/action_2/bias/Assign3^default_policy/sequential_3/action_2/kernel/Assign3^default_policy/sequential_3/action_out/bias/Assign5^default_policy/sequential_3/action_out/kernel/Assign3^default_policy/sequential_4/q_hidden_0/bias/Assign5^default_policy/sequential_4/q_hidden_0/kernel/Assign3^default_policy/sequential_4/q_hidden_1/bias/Assign5^default_policy/sequential_4/q_hidden_1/kernel/Assign.^default_policy/sequential_4/q_out/bias/Assign0^default_policy/sequential_4/q_out/kernel/Assign8^default_policy/sequential_5/twin_q_hidden_0/bias/Assign:^default_policy/sequential_5/twin_q_hidden_0/kernel/Assign8^default_policy/sequential_5/twin_q_hidden_1/bias/Assign:^default_policy/sequential_5/twin_q_hidden_1/kernel/Assign3^default_policy/sequential_5/twin_q_out/bias/Assign5^default_policy/sequential_5/twin_q_out/kernel/Assign%^default_policy/value_out/bias/Assign'^default_policy/value_out/kernel/Assign'^default_policy/value_out_1/bias/Assign)^default_policy/value_out_1/kernel/Assign
U
default_policy/arg_0Placeholder*
_output_shapes
: *
dtype0*
shape: 

$default_policy/mul_20/ReadVariableOpReadVariableOpdefault_policy/value_out/kernel*
_output_shapes
:	*
dtype0

default_policy/mul_20Muldefault_policy/arg_0$default_policy/mul_20/ReadVariableOp*
T0*
_output_shapes
:	
\
default_policy/sub_33/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
l
default_policy/sub_33Subdefault_policy/sub_33/xdefault_policy/arg_0*
T0*
_output_shapes
: 

$default_policy/mul_21/ReadVariableOpReadVariableOp!default_policy/value_out_1/kernel*
_output_shapes
:	*
dtype0

default_policy/mul_21Muldefault_policy/sub_33$default_policy/mul_21/ReadVariableOp*
T0*
_output_shapes
:	
v
default_policy/add_22AddV2default_policy/mul_20default_policy/mul_21*
T0*
_output_shapes
:	
Л
"default_policy/AssignVariableOp_42AssignVariableOp!default_policy/value_out_1/kerneldefault_policy/add_22*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Ј
 default_policy/ReadVariableOp_85ReadVariableOp!default_policy/value_out_1/kernel#^default_policy/AssignVariableOp_42*
_output_shapes
:	*
dtype0
~
$default_policy/mul_22/ReadVariableOpReadVariableOpdefault_policy/value_out/bias*
_output_shapes
:*
dtype0
}
default_policy/mul_22Muldefault_policy/arg_0$default_policy/mul_22/ReadVariableOp*
T0*
_output_shapes
:
\
default_policy/sub_34/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
l
default_policy/sub_34Subdefault_policy/sub_34/xdefault_policy/arg_0*
T0*
_output_shapes
: 

$default_policy/mul_23/ReadVariableOpReadVariableOpdefault_policy/value_out_1/bias*
_output_shapes
:*
dtype0
~
default_policy/mul_23Muldefault_policy/sub_34$default_policy/mul_23/ReadVariableOp*
T0*
_output_shapes
:
q
default_policy/add_23AddV2default_policy/mul_22default_policy/mul_23*
T0*
_output_shapes
:
Й
"default_policy/AssignVariableOp_43AssignVariableOpdefault_policy/value_out_1/biasdefault_policy/add_23*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Ё
 default_policy/ReadVariableOp_86ReadVariableOpdefault_policy/value_out_1/bias#^default_policy/AssignVariableOp_43*
_output_shapes
:*
dtype0

$default_policy/mul_24/ReadVariableOpReadVariableOp)default_policy/sequential/action_1/kernel* 
_output_shapes
:
*
dtype0

default_policy/mul_24Muldefault_policy/arg_0$default_policy/mul_24/ReadVariableOp*
T0* 
_output_shapes
:

\
default_policy/sub_35/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
l
default_policy/sub_35Subdefault_policy/sub_35/xdefault_policy/arg_0*
T0*
_output_shapes
: 

$default_policy/mul_25/ReadVariableOpReadVariableOp+default_policy/sequential_3/action_1/kernel* 
_output_shapes
:
*
dtype0

default_policy/mul_25Muldefault_policy/sub_35$default_policy/mul_25/ReadVariableOp*
T0* 
_output_shapes
:

w
default_policy/add_24AddV2default_policy/mul_24default_policy/mul_25*
T0* 
_output_shapes
:

Х
"default_policy/AssignVariableOp_44AssignVariableOp+default_policy/sequential_3/action_1/kerneldefault_policy/add_24*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Г
 default_policy/ReadVariableOp_87ReadVariableOp+default_policy/sequential_3/action_1/kernel#^default_policy/AssignVariableOp_44* 
_output_shapes
:
*
dtype0

$default_policy/mul_26/ReadVariableOpReadVariableOp'default_policy/sequential/action_1/bias*
_output_shapes	
:*
dtype0
~
default_policy/mul_26Muldefault_policy/arg_0$default_policy/mul_26/ReadVariableOp*
T0*
_output_shapes	
:
\
default_policy/sub_36/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
l
default_policy/sub_36Subdefault_policy/sub_36/xdefault_policy/arg_0*
T0*
_output_shapes
: 

$default_policy/mul_27/ReadVariableOpReadVariableOp)default_policy/sequential_3/action_1/bias*
_output_shapes	
:*
dtype0

default_policy/mul_27Muldefault_policy/sub_36$default_policy/mul_27/ReadVariableOp*
T0*
_output_shapes	
:
r
default_policy/add_25AddV2default_policy/mul_26default_policy/mul_27*
T0*
_output_shapes	
:
У
"default_policy/AssignVariableOp_45AssignVariableOp)default_policy/sequential_3/action_1/biasdefault_policy/add_25*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Ќ
 default_policy/ReadVariableOp_88ReadVariableOp)default_policy/sequential_3/action_1/bias#^default_policy/AssignVariableOp_45*
_output_shapes	
:*
dtype0

$default_policy/mul_28/ReadVariableOpReadVariableOp)default_policy/sequential/action_2/kernel* 
_output_shapes
:
*
dtype0

default_policy/mul_28Muldefault_policy/arg_0$default_policy/mul_28/ReadVariableOp*
T0* 
_output_shapes
:

\
default_policy/sub_37/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
l
default_policy/sub_37Subdefault_policy/sub_37/xdefault_policy/arg_0*
T0*
_output_shapes
: 

$default_policy/mul_29/ReadVariableOpReadVariableOp+default_policy/sequential_3/action_2/kernel* 
_output_shapes
:
*
dtype0

default_policy/mul_29Muldefault_policy/sub_37$default_policy/mul_29/ReadVariableOp*
T0* 
_output_shapes
:

w
default_policy/add_26AddV2default_policy/mul_28default_policy/mul_29*
T0* 
_output_shapes
:

Х
"default_policy/AssignVariableOp_46AssignVariableOp+default_policy/sequential_3/action_2/kerneldefault_policy/add_26*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Г
 default_policy/ReadVariableOp_89ReadVariableOp+default_policy/sequential_3/action_2/kernel#^default_policy/AssignVariableOp_46* 
_output_shapes
:
*
dtype0

$default_policy/mul_30/ReadVariableOpReadVariableOp'default_policy/sequential/action_2/bias*
_output_shapes	
:*
dtype0
~
default_policy/mul_30Muldefault_policy/arg_0$default_policy/mul_30/ReadVariableOp*
T0*
_output_shapes	
:
\
default_policy/sub_38/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
l
default_policy/sub_38Subdefault_policy/sub_38/xdefault_policy/arg_0*
T0*
_output_shapes
: 

$default_policy/mul_31/ReadVariableOpReadVariableOp)default_policy/sequential_3/action_2/bias*
_output_shapes	
:*
dtype0

default_policy/mul_31Muldefault_policy/sub_38$default_policy/mul_31/ReadVariableOp*
T0*
_output_shapes	
:
r
default_policy/add_27AddV2default_policy/mul_30default_policy/mul_31*
T0*
_output_shapes	
:
У
"default_policy/AssignVariableOp_47AssignVariableOp)default_policy/sequential_3/action_2/biasdefault_policy/add_27*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Ќ
 default_policy/ReadVariableOp_90ReadVariableOp)default_policy/sequential_3/action_2/bias#^default_policy/AssignVariableOp_47*
_output_shapes	
:*
dtype0

$default_policy/mul_32/ReadVariableOpReadVariableOp+default_policy/sequential/action_out/kernel*
_output_shapes
:	*
dtype0

default_policy/mul_32Muldefault_policy/arg_0$default_policy/mul_32/ReadVariableOp*
T0*
_output_shapes
:	
\
default_policy/sub_39/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
l
default_policy/sub_39Subdefault_policy/sub_39/xdefault_policy/arg_0*
T0*
_output_shapes
: 

$default_policy/mul_33/ReadVariableOpReadVariableOp-default_policy/sequential_3/action_out/kernel*
_output_shapes
:	*
dtype0

default_policy/mul_33Muldefault_policy/sub_39$default_policy/mul_33/ReadVariableOp*
T0*
_output_shapes
:	
v
default_policy/add_28AddV2default_policy/mul_32default_policy/mul_33*
T0*
_output_shapes
:	
Ч
"default_policy/AssignVariableOp_48AssignVariableOp-default_policy/sequential_3/action_out/kerneldefault_policy/add_28*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Д
 default_policy/ReadVariableOp_91ReadVariableOp-default_policy/sequential_3/action_out/kernel#^default_policy/AssignVariableOp_48*
_output_shapes
:	*
dtype0

$default_policy/mul_34/ReadVariableOpReadVariableOp)default_policy/sequential/action_out/bias*
_output_shapes
:*
dtype0
}
default_policy/mul_34Muldefault_policy/arg_0$default_policy/mul_34/ReadVariableOp*
T0*
_output_shapes
:
\
default_policy/sub_40/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
l
default_policy/sub_40Subdefault_policy/sub_40/xdefault_policy/arg_0*
T0*
_output_shapes
: 

$default_policy/mul_35/ReadVariableOpReadVariableOp+default_policy/sequential_3/action_out/bias*
_output_shapes
:*
dtype0
~
default_policy/mul_35Muldefault_policy/sub_40$default_policy/mul_35/ReadVariableOp*
T0*
_output_shapes
:
q
default_policy/add_29AddV2default_policy/mul_34default_policy/mul_35*
T0*
_output_shapes
:
Х
"default_policy/AssignVariableOp_49AssignVariableOp+default_policy/sequential_3/action_out/biasdefault_policy/add_29*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
­
 default_policy/ReadVariableOp_92ReadVariableOp+default_policy/sequential_3/action_out/bias#^default_policy/AssignVariableOp_49*
_output_shapes
:*
dtype0

$default_policy/mul_36/ReadVariableOpReadVariableOp-default_policy/sequential_1/q_hidden_0/kernel* 
_output_shapes
:
*
dtype0

default_policy/mul_36Muldefault_policy/arg_0$default_policy/mul_36/ReadVariableOp*
T0* 
_output_shapes
:

\
default_policy/sub_41/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
l
default_policy/sub_41Subdefault_policy/sub_41/xdefault_policy/arg_0*
T0*
_output_shapes
: 

$default_policy/mul_37/ReadVariableOpReadVariableOp-default_policy/sequential_4/q_hidden_0/kernel* 
_output_shapes
:
*
dtype0

default_policy/mul_37Muldefault_policy/sub_41$default_policy/mul_37/ReadVariableOp*
T0* 
_output_shapes
:

w
default_policy/add_30AddV2default_policy/mul_36default_policy/mul_37*
T0* 
_output_shapes
:

Ч
"default_policy/AssignVariableOp_50AssignVariableOp-default_policy/sequential_4/q_hidden_0/kerneldefault_policy/add_30*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Е
 default_policy/ReadVariableOp_93ReadVariableOp-default_policy/sequential_4/q_hidden_0/kernel#^default_policy/AssignVariableOp_50* 
_output_shapes
:
*
dtype0

$default_policy/mul_38/ReadVariableOpReadVariableOp+default_policy/sequential_1/q_hidden_0/bias*
_output_shapes	
:*
dtype0
~
default_policy/mul_38Muldefault_policy/arg_0$default_policy/mul_38/ReadVariableOp*
T0*
_output_shapes	
:
\
default_policy/sub_42/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
l
default_policy/sub_42Subdefault_policy/sub_42/xdefault_policy/arg_0*
T0*
_output_shapes
: 

$default_policy/mul_39/ReadVariableOpReadVariableOp+default_policy/sequential_4/q_hidden_0/bias*
_output_shapes	
:*
dtype0

default_policy/mul_39Muldefault_policy/sub_42$default_policy/mul_39/ReadVariableOp*
T0*
_output_shapes	
:
r
default_policy/add_31AddV2default_policy/mul_38default_policy/mul_39*
T0*
_output_shapes	
:
Х
"default_policy/AssignVariableOp_51AssignVariableOp+default_policy/sequential_4/q_hidden_0/biasdefault_policy/add_31*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Ў
 default_policy/ReadVariableOp_94ReadVariableOp+default_policy/sequential_4/q_hidden_0/bias#^default_policy/AssignVariableOp_51*
_output_shapes	
:*
dtype0

$default_policy/mul_40/ReadVariableOpReadVariableOp-default_policy/sequential_1/q_hidden_1/kernel* 
_output_shapes
:
*
dtype0

default_policy/mul_40Muldefault_policy/arg_0$default_policy/mul_40/ReadVariableOp*
T0* 
_output_shapes
:

\
default_policy/sub_43/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
l
default_policy/sub_43Subdefault_policy/sub_43/xdefault_policy/arg_0*
T0*
_output_shapes
: 

$default_policy/mul_41/ReadVariableOpReadVariableOp-default_policy/sequential_4/q_hidden_1/kernel* 
_output_shapes
:
*
dtype0

default_policy/mul_41Muldefault_policy/sub_43$default_policy/mul_41/ReadVariableOp*
T0* 
_output_shapes
:

w
default_policy/add_32AddV2default_policy/mul_40default_policy/mul_41*
T0* 
_output_shapes
:

Ч
"default_policy/AssignVariableOp_52AssignVariableOp-default_policy/sequential_4/q_hidden_1/kerneldefault_policy/add_32*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Е
 default_policy/ReadVariableOp_95ReadVariableOp-default_policy/sequential_4/q_hidden_1/kernel#^default_policy/AssignVariableOp_52* 
_output_shapes
:
*
dtype0

$default_policy/mul_42/ReadVariableOpReadVariableOp+default_policy/sequential_1/q_hidden_1/bias*
_output_shapes	
:*
dtype0
~
default_policy/mul_42Muldefault_policy/arg_0$default_policy/mul_42/ReadVariableOp*
T0*
_output_shapes	
:
\
default_policy/sub_44/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
l
default_policy/sub_44Subdefault_policy/sub_44/xdefault_policy/arg_0*
T0*
_output_shapes
: 

$default_policy/mul_43/ReadVariableOpReadVariableOp+default_policy/sequential_4/q_hidden_1/bias*
_output_shapes	
:*
dtype0

default_policy/mul_43Muldefault_policy/sub_44$default_policy/mul_43/ReadVariableOp*
T0*
_output_shapes	
:
r
default_policy/add_33AddV2default_policy/mul_42default_policy/mul_43*
T0*
_output_shapes	
:
Х
"default_policy/AssignVariableOp_53AssignVariableOp+default_policy/sequential_4/q_hidden_1/biasdefault_policy/add_33*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Ў
 default_policy/ReadVariableOp_96ReadVariableOp+default_policy/sequential_4/q_hidden_1/bias#^default_policy/AssignVariableOp_53*
_output_shapes	
:*
dtype0

$default_policy/mul_44/ReadVariableOpReadVariableOp(default_policy/sequential_1/q_out/kernel*
_output_shapes
:	*
dtype0

default_policy/mul_44Muldefault_policy/arg_0$default_policy/mul_44/ReadVariableOp*
T0*
_output_shapes
:	
\
default_policy/sub_45/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
l
default_policy/sub_45Subdefault_policy/sub_45/xdefault_policy/arg_0*
T0*
_output_shapes
: 

$default_policy/mul_45/ReadVariableOpReadVariableOp(default_policy/sequential_4/q_out/kernel*
_output_shapes
:	*
dtype0

default_policy/mul_45Muldefault_policy/sub_45$default_policy/mul_45/ReadVariableOp*
T0*
_output_shapes
:	
v
default_policy/add_34AddV2default_policy/mul_44default_policy/mul_45*
T0*
_output_shapes
:	
Т
"default_policy/AssignVariableOp_54AssignVariableOp(default_policy/sequential_4/q_out/kerneldefault_policy/add_34*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Џ
 default_policy/ReadVariableOp_97ReadVariableOp(default_policy/sequential_4/q_out/kernel#^default_policy/AssignVariableOp_54*
_output_shapes
:	*
dtype0

$default_policy/mul_46/ReadVariableOpReadVariableOp&default_policy/sequential_1/q_out/bias*
_output_shapes
:*
dtype0
}
default_policy/mul_46Muldefault_policy/arg_0$default_policy/mul_46/ReadVariableOp*
T0*
_output_shapes
:
\
default_policy/sub_46/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
l
default_policy/sub_46Subdefault_policy/sub_46/xdefault_policy/arg_0*
T0*
_output_shapes
: 

$default_policy/mul_47/ReadVariableOpReadVariableOp&default_policy/sequential_4/q_out/bias*
_output_shapes
:*
dtype0
~
default_policy/mul_47Muldefault_policy/sub_46$default_policy/mul_47/ReadVariableOp*
T0*
_output_shapes
:
q
default_policy/add_35AddV2default_policy/mul_46default_policy/mul_47*
T0*
_output_shapes
:
Р
"default_policy/AssignVariableOp_55AssignVariableOp&default_policy/sequential_4/q_out/biasdefault_policy/add_35*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Ј
 default_policy/ReadVariableOp_98ReadVariableOp&default_policy/sequential_4/q_out/bias#^default_policy/AssignVariableOp_55*
_output_shapes
:*
dtype0

$default_policy/mul_48/ReadVariableOpReadVariableOp2default_policy/sequential_2/twin_q_hidden_0/kernel* 
_output_shapes
:
*
dtype0

default_policy/mul_48Muldefault_policy/arg_0$default_policy/mul_48/ReadVariableOp*
T0* 
_output_shapes
:

\
default_policy/sub_47/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
l
default_policy/sub_47Subdefault_policy/sub_47/xdefault_policy/arg_0*
T0*
_output_shapes
: 

$default_policy/mul_49/ReadVariableOpReadVariableOp2default_policy/sequential_5/twin_q_hidden_0/kernel* 
_output_shapes
:
*
dtype0

default_policy/mul_49Muldefault_policy/sub_47$default_policy/mul_49/ReadVariableOp*
T0* 
_output_shapes
:

w
default_policy/add_36AddV2default_policy/mul_48default_policy/mul_49*
T0* 
_output_shapes
:

Ь
"default_policy/AssignVariableOp_56AssignVariableOp2default_policy/sequential_5/twin_q_hidden_0/kerneldefault_policy/add_36*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
К
 default_policy/ReadVariableOp_99ReadVariableOp2default_policy/sequential_5/twin_q_hidden_0/kernel#^default_policy/AssignVariableOp_56* 
_output_shapes
:
*
dtype0

$default_policy/mul_50/ReadVariableOpReadVariableOp0default_policy/sequential_2/twin_q_hidden_0/bias*
_output_shapes	
:*
dtype0
~
default_policy/mul_50Muldefault_policy/arg_0$default_policy/mul_50/ReadVariableOp*
T0*
_output_shapes	
:
\
default_policy/sub_48/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
l
default_policy/sub_48Subdefault_policy/sub_48/xdefault_policy/arg_0*
T0*
_output_shapes
: 

$default_policy/mul_51/ReadVariableOpReadVariableOp0default_policy/sequential_5/twin_q_hidden_0/bias*
_output_shapes	
:*
dtype0

default_policy/mul_51Muldefault_policy/sub_48$default_policy/mul_51/ReadVariableOp*
T0*
_output_shapes	
:
r
default_policy/add_37AddV2default_policy/mul_50default_policy/mul_51*
T0*
_output_shapes	
:
Ъ
"default_policy/AssignVariableOp_57AssignVariableOp0default_policy/sequential_5/twin_q_hidden_0/biasdefault_policy/add_37*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Д
!default_policy/ReadVariableOp_100ReadVariableOp0default_policy/sequential_5/twin_q_hidden_0/bias#^default_policy/AssignVariableOp_57*
_output_shapes	
:*
dtype0

$default_policy/mul_52/ReadVariableOpReadVariableOp2default_policy/sequential_2/twin_q_hidden_1/kernel* 
_output_shapes
:
*
dtype0

default_policy/mul_52Muldefault_policy/arg_0$default_policy/mul_52/ReadVariableOp*
T0* 
_output_shapes
:

\
default_policy/sub_49/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
l
default_policy/sub_49Subdefault_policy/sub_49/xdefault_policy/arg_0*
T0*
_output_shapes
: 

$default_policy/mul_53/ReadVariableOpReadVariableOp2default_policy/sequential_5/twin_q_hidden_1/kernel* 
_output_shapes
:
*
dtype0

default_policy/mul_53Muldefault_policy/sub_49$default_policy/mul_53/ReadVariableOp*
T0* 
_output_shapes
:

w
default_policy/add_38AddV2default_policy/mul_52default_policy/mul_53*
T0* 
_output_shapes
:

Ь
"default_policy/AssignVariableOp_58AssignVariableOp2default_policy/sequential_5/twin_q_hidden_1/kerneldefault_policy/add_38*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Л
!default_policy/ReadVariableOp_101ReadVariableOp2default_policy/sequential_5/twin_q_hidden_1/kernel#^default_policy/AssignVariableOp_58* 
_output_shapes
:
*
dtype0

$default_policy/mul_54/ReadVariableOpReadVariableOp0default_policy/sequential_2/twin_q_hidden_1/bias*
_output_shapes	
:*
dtype0
~
default_policy/mul_54Muldefault_policy/arg_0$default_policy/mul_54/ReadVariableOp*
T0*
_output_shapes	
:
\
default_policy/sub_50/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
l
default_policy/sub_50Subdefault_policy/sub_50/xdefault_policy/arg_0*
T0*
_output_shapes
: 

$default_policy/mul_55/ReadVariableOpReadVariableOp0default_policy/sequential_5/twin_q_hidden_1/bias*
_output_shapes	
:*
dtype0

default_policy/mul_55Muldefault_policy/sub_50$default_policy/mul_55/ReadVariableOp*
T0*
_output_shapes	
:
r
default_policy/add_39AddV2default_policy/mul_54default_policy/mul_55*
T0*
_output_shapes	
:
Ъ
"default_policy/AssignVariableOp_59AssignVariableOp0default_policy/sequential_5/twin_q_hidden_1/biasdefault_policy/add_39*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Д
!default_policy/ReadVariableOp_102ReadVariableOp0default_policy/sequential_5/twin_q_hidden_1/bias#^default_policy/AssignVariableOp_59*
_output_shapes	
:*
dtype0

$default_policy/mul_56/ReadVariableOpReadVariableOp-default_policy/sequential_2/twin_q_out/kernel*
_output_shapes
:	*
dtype0

default_policy/mul_56Muldefault_policy/arg_0$default_policy/mul_56/ReadVariableOp*
T0*
_output_shapes
:	
\
default_policy/sub_51/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
l
default_policy/sub_51Subdefault_policy/sub_51/xdefault_policy/arg_0*
T0*
_output_shapes
: 

$default_policy/mul_57/ReadVariableOpReadVariableOp-default_policy/sequential_5/twin_q_out/kernel*
_output_shapes
:	*
dtype0

default_policy/mul_57Muldefault_policy/sub_51$default_policy/mul_57/ReadVariableOp*
T0*
_output_shapes
:	
v
default_policy/add_40AddV2default_policy/mul_56default_policy/mul_57*
T0*
_output_shapes
:	
Ч
"default_policy/AssignVariableOp_60AssignVariableOp-default_policy/sequential_5/twin_q_out/kerneldefault_policy/add_40*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Е
!default_policy/ReadVariableOp_103ReadVariableOp-default_policy/sequential_5/twin_q_out/kernel#^default_policy/AssignVariableOp_60*
_output_shapes
:	*
dtype0

$default_policy/mul_58/ReadVariableOpReadVariableOp+default_policy/sequential_2/twin_q_out/bias*
_output_shapes
:*
dtype0
}
default_policy/mul_58Muldefault_policy/arg_0$default_policy/mul_58/ReadVariableOp*
T0*
_output_shapes
:
\
default_policy/sub_52/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
l
default_policy/sub_52Subdefault_policy/sub_52/xdefault_policy/arg_0*
T0*
_output_shapes
: 

$default_policy/mul_59/ReadVariableOpReadVariableOp+default_policy/sequential_5/twin_q_out/bias*
_output_shapes
:*
dtype0
~
default_policy/mul_59Muldefault_policy/sub_52$default_policy/mul_59/ReadVariableOp*
T0*
_output_shapes
:
q
default_policy/add_41AddV2default_policy/mul_58default_policy/mul_59*
T0*
_output_shapes
:
Х
"default_policy/AssignVariableOp_61AssignVariableOp+default_policy/sequential_5/twin_q_out/biasdefault_policy/add_41*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(
Ў
!default_policy/ReadVariableOp_104ReadVariableOp+default_policy/sequential_5/twin_q_out/bias#^default_policy/AssignVariableOp_61*
_output_shapes
:*
dtype0
u
$default_policy/mul_60/ReadVariableOpReadVariableOpdefault_policy/log_alpha*
_output_shapes
: *
dtype0
y
default_policy/mul_60Muldefault_policy/arg_0$default_policy/mul_60/ReadVariableOp*
T0*
_output_shapes
: 
\
default_policy/sub_53/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
l
default_policy/sub_53Subdefault_policy/sub_53/xdefault_policy/arg_0*
T0*
_output_shapes
: 
w
$default_policy/mul_61/ReadVariableOpReadVariableOpdefault_policy/log_alpha_1*
_output_shapes
: *
dtype0
z
default_policy/mul_61Muldefault_policy/sub_53$default_policy/mul_61/ReadVariableOp*
T0*
_output_shapes
: 
m
default_policy/add_42AddV2default_policy/mul_60default_policy/mul_61*
T0*
_output_shapes
: 
Д
"default_policy/AssignVariableOp_62AssignVariableOpdefault_policy/log_alpha_1default_policy/add_42*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape(

!default_policy/ReadVariableOp_105ReadVariableOpdefault_policy/log_alpha_1#^default_policy/AssignVariableOp_62*
_output_shapes
: *
dtype0
Ќ
default_policy/group_deps_1NoOp#^default_policy/AssignVariableOp_42#^default_policy/AssignVariableOp_43#^default_policy/AssignVariableOp_44#^default_policy/AssignVariableOp_45#^default_policy/AssignVariableOp_46#^default_policy/AssignVariableOp_47#^default_policy/AssignVariableOp_48#^default_policy/AssignVariableOp_49#^default_policy/AssignVariableOp_50#^default_policy/AssignVariableOp_51#^default_policy/AssignVariableOp_52#^default_policy/AssignVariableOp_53#^default_policy/AssignVariableOp_54#^default_policy/AssignVariableOp_55#^default_policy/AssignVariableOp_56#^default_policy/AssignVariableOp_57#^default_policy/AssignVariableOp_58#^default_policy/AssignVariableOp_59#^default_policy/AssignVariableOp_60#^default_policy/AssignVariableOp_61#^default_policy/AssignVariableOp_62
Y
save/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 
{
save/StaticRegexFullMatchStaticRegexFullMatch
save/Const"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
a
save/Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
f
save/Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
|
save/SelectSelectsave/StaticRegexFullMatchsave/Const_1save/Const_2"/device:CPU:**
T0*
_output_shapes
: 
w
save/StringJoin
StringJoin
save/Constsave/Select"/device:CPU:**
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
й$
save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*§#
valueѓ#B№#YBdefault_policy/beta1_powerBdefault_policy/beta1_power_1Bdefault_policy/beta1_power_2Bdefault_policy/beta1_power_3Bdefault_policy/beta2_powerBdefault_policy/beta2_power_1Bdefault_policy/beta2_power_2Bdefault_policy/beta2_power_3B,default_policy/default_policy/log_alpha/AdamB.default_policy/default_policy/log_alpha/Adam_1B;default_policy/default_policy/sequential/action_1/bias/AdamB=default_policy/default_policy/sequential/action_1/bias/Adam_1B=default_policy/default_policy/sequential/action_1/kernel/AdamB?default_policy/default_policy/sequential/action_1/kernel/Adam_1B;default_policy/default_policy/sequential/action_2/bias/AdamB=default_policy/default_policy/sequential/action_2/bias/Adam_1B=default_policy/default_policy/sequential/action_2/kernel/AdamB?default_policy/default_policy/sequential/action_2/kernel/Adam_1B=default_policy/default_policy/sequential/action_out/bias/AdamB?default_policy/default_policy/sequential/action_out/bias/Adam_1B?default_policy/default_policy/sequential/action_out/kernel/AdamBAdefault_policy/default_policy/sequential/action_out/kernel/Adam_1B?default_policy/default_policy/sequential_1/q_hidden_0/bias/AdamBAdefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam_1BAdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/AdamBCdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam_1B?default_policy/default_policy/sequential_1/q_hidden_1/bias/AdamBAdefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam_1BAdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/AdamBCdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam_1B:default_policy/default_policy/sequential_1/q_out/bias/AdamB<default_policy/default_policy/sequential_1/q_out/bias/Adam_1B<default_policy/default_policy/sequential_1/q_out/kernel/AdamB>default_policy/default_policy/sequential_1/q_out/kernel/Adam_1BDdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/AdamBFdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam_1BFdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/AdamBHdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam_1BDdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/AdamBFdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam_1BFdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/AdamBHdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam_1B?default_policy/default_policy/sequential_2/twin_q_out/bias/AdamBAdefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam_1BAdefault_policy/default_policy/sequential_2/twin_q_out/kernel/AdamBCdefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam_1Bdefault_policy/global_stepBdefault_policy/log_alphaBdefault_policy/log_alpha_1B'default_policy/sequential/action_1/biasB)default_policy/sequential/action_1/kernelB'default_policy/sequential/action_2/biasB)default_policy/sequential/action_2/kernelB)default_policy/sequential/action_out/biasB+default_policy/sequential/action_out/kernelB+default_policy/sequential_1/q_hidden_0/biasB-default_policy/sequential_1/q_hidden_0/kernelB+default_policy/sequential_1/q_hidden_1/biasB-default_policy/sequential_1/q_hidden_1/kernelB&default_policy/sequential_1/q_out/biasB(default_policy/sequential_1/q_out/kernelB0default_policy/sequential_2/twin_q_hidden_0/biasB2default_policy/sequential_2/twin_q_hidden_0/kernelB0default_policy/sequential_2/twin_q_hidden_1/biasB2default_policy/sequential_2/twin_q_hidden_1/kernelB+default_policy/sequential_2/twin_q_out/biasB-default_policy/sequential_2/twin_q_out/kernelB)default_policy/sequential_3/action_1/biasB+default_policy/sequential_3/action_1/kernelB)default_policy/sequential_3/action_2/biasB+default_policy/sequential_3/action_2/kernelB+default_policy/sequential_3/action_out/biasB-default_policy/sequential_3/action_out/kernelB+default_policy/sequential_4/q_hidden_0/biasB-default_policy/sequential_4/q_hidden_0/kernelB+default_policy/sequential_4/q_hidden_1/biasB-default_policy/sequential_4/q_hidden_1/kernelB&default_policy/sequential_4/q_out/biasB(default_policy/sequential_4/q_out/kernelB0default_policy/sequential_5/twin_q_hidden_0/biasB2default_policy/sequential_5/twin_q_hidden_0/kernelB0default_policy/sequential_5/twin_q_hidden_1/biasB2default_policy/sequential_5/twin_q_hidden_1/kernelB+default_policy/sequential_5/twin_q_out/biasB-default_policy/sequential_5/twin_q_out/kernelBdefault_policy/value_out/biasBdefault_policy/value_out/kernelBdefault_policy/value_out_1/biasB!default_policy/value_out_1/kernel
Ї
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*Ч
valueНBКYB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Ы3
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices.default_policy/beta1_power/Read/ReadVariableOp0default_policy/beta1_power_1/Read/ReadVariableOp0default_policy/beta1_power_2/Read/ReadVariableOp0default_policy/beta1_power_3/Read/ReadVariableOp.default_policy/beta2_power/Read/ReadVariableOp0default_policy/beta2_power_1/Read/ReadVariableOp0default_policy/beta2_power_2/Read/ReadVariableOp0default_policy/beta2_power_3/Read/ReadVariableOp@default_policy/default_policy/log_alpha/Adam/Read/ReadVariableOpBdefault_policy/default_policy/log_alpha/Adam_1/Read/ReadVariableOpOdefault_policy/default_policy/sequential/action_1/bias/Adam/Read/ReadVariableOpQdefault_policy/default_policy/sequential/action_1/bias/Adam_1/Read/ReadVariableOpQdefault_policy/default_policy/sequential/action_1/kernel/Adam/Read/ReadVariableOpSdefault_policy/default_policy/sequential/action_1/kernel/Adam_1/Read/ReadVariableOpOdefault_policy/default_policy/sequential/action_2/bias/Adam/Read/ReadVariableOpQdefault_policy/default_policy/sequential/action_2/bias/Adam_1/Read/ReadVariableOpQdefault_policy/default_policy/sequential/action_2/kernel/Adam/Read/ReadVariableOpSdefault_policy/default_policy/sequential/action_2/kernel/Adam_1/Read/ReadVariableOpQdefault_policy/default_policy/sequential/action_out/bias/Adam/Read/ReadVariableOpSdefault_policy/default_policy/sequential/action_out/bias/Adam_1/Read/ReadVariableOpSdefault_policy/default_policy/sequential/action_out/kernel/Adam/Read/ReadVariableOpUdefault_policy/default_policy/sequential/action_out/kernel/Adam_1/Read/ReadVariableOpSdefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam/Read/ReadVariableOpUdefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam_1/Read/ReadVariableOpUdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam/Read/ReadVariableOpWdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam_1/Read/ReadVariableOpSdefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam/Read/ReadVariableOpUdefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam_1/Read/ReadVariableOpUdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam/Read/ReadVariableOpWdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam_1/Read/ReadVariableOpNdefault_policy/default_policy/sequential_1/q_out/bias/Adam/Read/ReadVariableOpPdefault_policy/default_policy/sequential_1/q_out/bias/Adam_1/Read/ReadVariableOpPdefault_policy/default_policy/sequential_1/q_out/kernel/Adam/Read/ReadVariableOpRdefault_policy/default_policy/sequential_1/q_out/kernel/Adam_1/Read/ReadVariableOpXdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam/Read/ReadVariableOpZdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam_1/Read/ReadVariableOpZdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam/Read/ReadVariableOp\default_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam_1/Read/ReadVariableOpXdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam/Read/ReadVariableOpZdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam_1/Read/ReadVariableOpZdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam/Read/ReadVariableOp\default_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam_1/Read/ReadVariableOpSdefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam/Read/ReadVariableOpUdefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam_1/Read/ReadVariableOpUdefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam/Read/ReadVariableOpWdefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam_1/Read/ReadVariableOpdefault_policy/global_step,default_policy/log_alpha/Read/ReadVariableOp.default_policy/log_alpha_1/Read/ReadVariableOp;default_policy/sequential/action_1/bias/Read/ReadVariableOp=default_policy/sequential/action_1/kernel/Read/ReadVariableOp;default_policy/sequential/action_2/bias/Read/ReadVariableOp=default_policy/sequential/action_2/kernel/Read/ReadVariableOp=default_policy/sequential/action_out/bias/Read/ReadVariableOp?default_policy/sequential/action_out/kernel/Read/ReadVariableOp?default_policy/sequential_1/q_hidden_0/bias/Read/ReadVariableOpAdefault_policy/sequential_1/q_hidden_0/kernel/Read/ReadVariableOp?default_policy/sequential_1/q_hidden_1/bias/Read/ReadVariableOpAdefault_policy/sequential_1/q_hidden_1/kernel/Read/ReadVariableOp:default_policy/sequential_1/q_out/bias/Read/ReadVariableOp<default_policy/sequential_1/q_out/kernel/Read/ReadVariableOpDdefault_policy/sequential_2/twin_q_hidden_0/bias/Read/ReadVariableOpFdefault_policy/sequential_2/twin_q_hidden_0/kernel/Read/ReadVariableOpDdefault_policy/sequential_2/twin_q_hidden_1/bias/Read/ReadVariableOpFdefault_policy/sequential_2/twin_q_hidden_1/kernel/Read/ReadVariableOp?default_policy/sequential_2/twin_q_out/bias/Read/ReadVariableOpAdefault_policy/sequential_2/twin_q_out/kernel/Read/ReadVariableOp=default_policy/sequential_3/action_1/bias/Read/ReadVariableOp?default_policy/sequential_3/action_1/kernel/Read/ReadVariableOp=default_policy/sequential_3/action_2/bias/Read/ReadVariableOp?default_policy/sequential_3/action_2/kernel/Read/ReadVariableOp?default_policy/sequential_3/action_out/bias/Read/ReadVariableOpAdefault_policy/sequential_3/action_out/kernel/Read/ReadVariableOp?default_policy/sequential_4/q_hidden_0/bias/Read/ReadVariableOpAdefault_policy/sequential_4/q_hidden_0/kernel/Read/ReadVariableOp?default_policy/sequential_4/q_hidden_1/bias/Read/ReadVariableOpAdefault_policy/sequential_4/q_hidden_1/kernel/Read/ReadVariableOp:default_policy/sequential_4/q_out/bias/Read/ReadVariableOp<default_policy/sequential_4/q_out/kernel/Read/ReadVariableOpDdefault_policy/sequential_5/twin_q_hidden_0/bias/Read/ReadVariableOpFdefault_policy/sequential_5/twin_q_hidden_0/kernel/Read/ReadVariableOpDdefault_policy/sequential_5/twin_q_hidden_1/bias/Read/ReadVariableOpFdefault_policy/sequential_5/twin_q_hidden_1/kernel/Read/ReadVariableOp?default_policy/sequential_5/twin_q_out/bias/Read/ReadVariableOpAdefault_policy/sequential_5/twin_q_out/kernel/Read/ReadVariableOp1default_policy/value_out/bias/Read/ReadVariableOp3default_policy/value_out/kernel/Read/ReadVariableOp3default_policy/value_out_1/bias/Read/ReadVariableOp5default_policy/value_out_1/kernel/Read/ReadVariableOp"/device:CPU:0*&
 _has_manual_control_dependencies(*g
dtypes]
[2Y	
Ш
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*&
 _has_manual_control_dependencies(*
_output_shapes
: 
Ќ
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
T0*
_output_shapes
:*

axis 
Я
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*&
 _has_manual_control_dependencies(*
allow_missing_files( *
delete_old_dirs(

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
м$
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*§#
valueѓ#B№#YBdefault_policy/beta1_powerBdefault_policy/beta1_power_1Bdefault_policy/beta1_power_2Bdefault_policy/beta1_power_3Bdefault_policy/beta2_powerBdefault_policy/beta2_power_1Bdefault_policy/beta2_power_2Bdefault_policy/beta2_power_3B,default_policy/default_policy/log_alpha/AdamB.default_policy/default_policy/log_alpha/Adam_1B;default_policy/default_policy/sequential/action_1/bias/AdamB=default_policy/default_policy/sequential/action_1/bias/Adam_1B=default_policy/default_policy/sequential/action_1/kernel/AdamB?default_policy/default_policy/sequential/action_1/kernel/Adam_1B;default_policy/default_policy/sequential/action_2/bias/AdamB=default_policy/default_policy/sequential/action_2/bias/Adam_1B=default_policy/default_policy/sequential/action_2/kernel/AdamB?default_policy/default_policy/sequential/action_2/kernel/Adam_1B=default_policy/default_policy/sequential/action_out/bias/AdamB?default_policy/default_policy/sequential/action_out/bias/Adam_1B?default_policy/default_policy/sequential/action_out/kernel/AdamBAdefault_policy/default_policy/sequential/action_out/kernel/Adam_1B?default_policy/default_policy/sequential_1/q_hidden_0/bias/AdamBAdefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam_1BAdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/AdamBCdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam_1B?default_policy/default_policy/sequential_1/q_hidden_1/bias/AdamBAdefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam_1BAdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/AdamBCdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam_1B:default_policy/default_policy/sequential_1/q_out/bias/AdamB<default_policy/default_policy/sequential_1/q_out/bias/Adam_1B<default_policy/default_policy/sequential_1/q_out/kernel/AdamB>default_policy/default_policy/sequential_1/q_out/kernel/Adam_1BDdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/AdamBFdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam_1BFdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/AdamBHdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam_1BDdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/AdamBFdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam_1BFdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/AdamBHdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam_1B?default_policy/default_policy/sequential_2/twin_q_out/bias/AdamBAdefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam_1BAdefault_policy/default_policy/sequential_2/twin_q_out/kernel/AdamBCdefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam_1Bdefault_policy/global_stepBdefault_policy/log_alphaBdefault_policy/log_alpha_1B'default_policy/sequential/action_1/biasB)default_policy/sequential/action_1/kernelB'default_policy/sequential/action_2/biasB)default_policy/sequential/action_2/kernelB)default_policy/sequential/action_out/biasB+default_policy/sequential/action_out/kernelB+default_policy/sequential_1/q_hidden_0/biasB-default_policy/sequential_1/q_hidden_0/kernelB+default_policy/sequential_1/q_hidden_1/biasB-default_policy/sequential_1/q_hidden_1/kernelB&default_policy/sequential_1/q_out/biasB(default_policy/sequential_1/q_out/kernelB0default_policy/sequential_2/twin_q_hidden_0/biasB2default_policy/sequential_2/twin_q_hidden_0/kernelB0default_policy/sequential_2/twin_q_hidden_1/biasB2default_policy/sequential_2/twin_q_hidden_1/kernelB+default_policy/sequential_2/twin_q_out/biasB-default_policy/sequential_2/twin_q_out/kernelB)default_policy/sequential_3/action_1/biasB+default_policy/sequential_3/action_1/kernelB)default_policy/sequential_3/action_2/biasB+default_policy/sequential_3/action_2/kernelB+default_policy/sequential_3/action_out/biasB-default_policy/sequential_3/action_out/kernelB+default_policy/sequential_4/q_hidden_0/biasB-default_policy/sequential_4/q_hidden_0/kernelB+default_policy/sequential_4/q_hidden_1/biasB-default_policy/sequential_4/q_hidden_1/kernelB&default_policy/sequential_4/q_out/biasB(default_policy/sequential_4/q_out/kernelB0default_policy/sequential_5/twin_q_hidden_0/biasB2default_policy/sequential_5/twin_q_hidden_0/kernelB0default_policy/sequential_5/twin_q_hidden_1/biasB2default_policy/sequential_5/twin_q_hidden_1/kernelB+default_policy/sequential_5/twin_q_out/biasB-default_policy/sequential_5/twin_q_out/kernelBdefault_policy/value_out/biasBdefault_policy/value_out/kernelBdefault_policy/value_out_1/biasB!default_policy/value_out_1/kernel
Њ
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*Ч
valueНBКYB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
к
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*њ
_output_shapesч
ф:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*g
dtypes]
[2Y	
N
save/Identity_1Identitysave/RestoreV2*
T0*
_output_shapes
:
Ё
save/AssignVariableOpAssignVariableOpdefault_policy/beta1_powersave/Identity_1*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
P
save/Identity_2Identitysave/RestoreV2:1*
T0*
_output_shapes
:
Ѕ
save/AssignVariableOp_1AssignVariableOpdefault_policy/beta1_power_1save/Identity_2*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
P
save/Identity_3Identitysave/RestoreV2:2*
T0*
_output_shapes
:
Ѕ
save/AssignVariableOp_2AssignVariableOpdefault_policy/beta1_power_2save/Identity_3*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
P
save/Identity_4Identitysave/RestoreV2:3*
T0*
_output_shapes
:
Ѕ
save/AssignVariableOp_3AssignVariableOpdefault_policy/beta1_power_3save/Identity_4*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
P
save/Identity_5Identitysave/RestoreV2:4*
T0*
_output_shapes
:
Ѓ
save/AssignVariableOp_4AssignVariableOpdefault_policy/beta2_powersave/Identity_5*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
P
save/Identity_6Identitysave/RestoreV2:5*
T0*
_output_shapes
:
Ѕ
save/AssignVariableOp_5AssignVariableOpdefault_policy/beta2_power_1save/Identity_6*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
P
save/Identity_7Identitysave/RestoreV2:6*
T0*
_output_shapes
:
Ѕ
save/AssignVariableOp_6AssignVariableOpdefault_policy/beta2_power_2save/Identity_7*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
P
save/Identity_8Identitysave/RestoreV2:7*
T0*
_output_shapes
:
Ѕ
save/AssignVariableOp_7AssignVariableOpdefault_policy/beta2_power_3save/Identity_8*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
P
save/Identity_9Identitysave/RestoreV2:8*
T0*
_output_shapes
:
Е
save/AssignVariableOp_8AssignVariableOp,default_policy/default_policy/log_alpha/Adamsave/Identity_9*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
Q
save/Identity_10Identitysave/RestoreV2:9*
T0*
_output_shapes
:
И
save/AssignVariableOp_9AssignVariableOp.default_policy/default_policy/log_alpha/Adam_1save/Identity_10*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_11Identitysave/RestoreV2:10*
T0*
_output_shapes
:
Ц
save/AssignVariableOp_10AssignVariableOp;default_policy/default_policy/sequential/action_1/bias/Adamsave/Identity_11*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_12Identitysave/RestoreV2:11*
T0*
_output_shapes
:
Ш
save/AssignVariableOp_11AssignVariableOp=default_policy/default_policy/sequential/action_1/bias/Adam_1save/Identity_12*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_13Identitysave/RestoreV2:12*
T0*
_output_shapes
:
Ш
save/AssignVariableOp_12AssignVariableOp=default_policy/default_policy/sequential/action_1/kernel/Adamsave/Identity_13*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_14Identitysave/RestoreV2:13*
T0*
_output_shapes
:
Ъ
save/AssignVariableOp_13AssignVariableOp?default_policy/default_policy/sequential/action_1/kernel/Adam_1save/Identity_14*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_15Identitysave/RestoreV2:14*
T0*
_output_shapes
:
Ц
save/AssignVariableOp_14AssignVariableOp;default_policy/default_policy/sequential/action_2/bias/Adamsave/Identity_15*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_16Identitysave/RestoreV2:15*
T0*
_output_shapes
:
Ш
save/AssignVariableOp_15AssignVariableOp=default_policy/default_policy/sequential/action_2/bias/Adam_1save/Identity_16*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_17Identitysave/RestoreV2:16*
T0*
_output_shapes
:
Ш
save/AssignVariableOp_16AssignVariableOp=default_policy/default_policy/sequential/action_2/kernel/Adamsave/Identity_17*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_18Identitysave/RestoreV2:17*
T0*
_output_shapes
:
Ъ
save/AssignVariableOp_17AssignVariableOp?default_policy/default_policy/sequential/action_2/kernel/Adam_1save/Identity_18*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_19Identitysave/RestoreV2:18*
T0*
_output_shapes
:
Ш
save/AssignVariableOp_18AssignVariableOp=default_policy/default_policy/sequential/action_out/bias/Adamsave/Identity_19*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_20Identitysave/RestoreV2:19*
T0*
_output_shapes
:
Ъ
save/AssignVariableOp_19AssignVariableOp?default_policy/default_policy/sequential/action_out/bias/Adam_1save/Identity_20*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_21Identitysave/RestoreV2:20*
T0*
_output_shapes
:
Ъ
save/AssignVariableOp_20AssignVariableOp?default_policy/default_policy/sequential/action_out/kernel/Adamsave/Identity_21*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_22Identitysave/RestoreV2:21*
T0*
_output_shapes
:
Ь
save/AssignVariableOp_21AssignVariableOpAdefault_policy/default_policy/sequential/action_out/kernel/Adam_1save/Identity_22*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_23Identitysave/RestoreV2:22*
T0*
_output_shapes
:
Ъ
save/AssignVariableOp_22AssignVariableOp?default_policy/default_policy/sequential_1/q_hidden_0/bias/Adamsave/Identity_23*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_24Identitysave/RestoreV2:23*
T0*
_output_shapes
:
Ь
save/AssignVariableOp_23AssignVariableOpAdefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam_1save/Identity_24*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_25Identitysave/RestoreV2:24*
T0*
_output_shapes
:
Ь
save/AssignVariableOp_24AssignVariableOpAdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adamsave/Identity_25*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_26Identitysave/RestoreV2:25*
T0*
_output_shapes
:
Ю
save/AssignVariableOp_25AssignVariableOpCdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam_1save/Identity_26*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_27Identitysave/RestoreV2:26*
T0*
_output_shapes
:
Ъ
save/AssignVariableOp_26AssignVariableOp?default_policy/default_policy/sequential_1/q_hidden_1/bias/Adamsave/Identity_27*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_28Identitysave/RestoreV2:27*
T0*
_output_shapes
:
Ь
save/AssignVariableOp_27AssignVariableOpAdefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam_1save/Identity_28*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_29Identitysave/RestoreV2:28*
T0*
_output_shapes
:
Ь
save/AssignVariableOp_28AssignVariableOpAdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adamsave/Identity_29*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_30Identitysave/RestoreV2:29*
T0*
_output_shapes
:
Ю
save/AssignVariableOp_29AssignVariableOpCdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam_1save/Identity_30*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_31Identitysave/RestoreV2:30*
T0*
_output_shapes
:
Х
save/AssignVariableOp_30AssignVariableOp:default_policy/default_policy/sequential_1/q_out/bias/Adamsave/Identity_31*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_32Identitysave/RestoreV2:31*
T0*
_output_shapes
:
Ч
save/AssignVariableOp_31AssignVariableOp<default_policy/default_policy/sequential_1/q_out/bias/Adam_1save/Identity_32*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_33Identitysave/RestoreV2:32*
T0*
_output_shapes
:
Ч
save/AssignVariableOp_32AssignVariableOp<default_policy/default_policy/sequential_1/q_out/kernel/Adamsave/Identity_33*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_34Identitysave/RestoreV2:33*
T0*
_output_shapes
:
Щ
save/AssignVariableOp_33AssignVariableOp>default_policy/default_policy/sequential_1/q_out/kernel/Adam_1save/Identity_34*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_35Identitysave/RestoreV2:34*
T0*
_output_shapes
:
Я
save/AssignVariableOp_34AssignVariableOpDdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adamsave/Identity_35*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_36Identitysave/RestoreV2:35*
T0*
_output_shapes
:
б
save/AssignVariableOp_35AssignVariableOpFdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam_1save/Identity_36*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_37Identitysave/RestoreV2:36*
T0*
_output_shapes
:
б
save/AssignVariableOp_36AssignVariableOpFdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adamsave/Identity_37*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_38Identitysave/RestoreV2:37*
T0*
_output_shapes
:
г
save/AssignVariableOp_37AssignVariableOpHdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam_1save/Identity_38*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_39Identitysave/RestoreV2:38*
T0*
_output_shapes
:
Я
save/AssignVariableOp_38AssignVariableOpDdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adamsave/Identity_39*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_40Identitysave/RestoreV2:39*
T0*
_output_shapes
:
б
save/AssignVariableOp_39AssignVariableOpFdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam_1save/Identity_40*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_41Identitysave/RestoreV2:40*
T0*
_output_shapes
:
б
save/AssignVariableOp_40AssignVariableOpFdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adamsave/Identity_41*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_42Identitysave/RestoreV2:41*
T0*
_output_shapes
:
г
save/AssignVariableOp_41AssignVariableOpHdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam_1save/Identity_42*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_43Identitysave/RestoreV2:42*
T0*
_output_shapes
:
Ъ
save/AssignVariableOp_42AssignVariableOp?default_policy/default_policy/sequential_2/twin_q_out/bias/Adamsave/Identity_43*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_44Identitysave/RestoreV2:43*
T0*
_output_shapes
:
Ь
save/AssignVariableOp_43AssignVariableOpAdefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam_1save/Identity_44*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_45Identitysave/RestoreV2:44*
T0*
_output_shapes
:
Ь
save/AssignVariableOp_44AssignVariableOpAdefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adamsave/Identity_45*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_46Identitysave/RestoreV2:45*
T0*
_output_shapes
:
Ю
save/AssignVariableOp_45AssignVariableOpCdefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam_1save/Identity_46*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
х
save/AssignAssigndefault_policy/global_stepsave/RestoreV2:46*
T0	*-
_class#
!loc:@default_policy/global_step*&
 _has_manual_control_dependencies(*
_output_shapes
: *
use_locking(*
validate_shape(
R
save/Identity_47Identitysave/RestoreV2:47*
T0*
_output_shapes
:
Ѓ
save/AssignVariableOp_46AssignVariableOpdefault_policy/log_alphasave/Identity_47*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_48Identitysave/RestoreV2:48*
T0*
_output_shapes
:
Ѕ
save/AssignVariableOp_47AssignVariableOpdefault_policy/log_alpha_1save/Identity_48*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_49Identitysave/RestoreV2:49*
T0*
_output_shapes
:
В
save/AssignVariableOp_48AssignVariableOp'default_policy/sequential/action_1/biassave/Identity_49*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_50Identitysave/RestoreV2:50*
T0*
_output_shapes
:
Д
save/AssignVariableOp_49AssignVariableOp)default_policy/sequential/action_1/kernelsave/Identity_50*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_51Identitysave/RestoreV2:51*
T0*
_output_shapes
:
В
save/AssignVariableOp_50AssignVariableOp'default_policy/sequential/action_2/biassave/Identity_51*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_52Identitysave/RestoreV2:52*
T0*
_output_shapes
:
Д
save/AssignVariableOp_51AssignVariableOp)default_policy/sequential/action_2/kernelsave/Identity_52*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_53Identitysave/RestoreV2:53*
T0*
_output_shapes
:
Д
save/AssignVariableOp_52AssignVariableOp)default_policy/sequential/action_out/biassave/Identity_53*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_54Identitysave/RestoreV2:54*
T0*
_output_shapes
:
Ж
save/AssignVariableOp_53AssignVariableOp+default_policy/sequential/action_out/kernelsave/Identity_54*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_55Identitysave/RestoreV2:55*
T0*
_output_shapes
:
Ж
save/AssignVariableOp_54AssignVariableOp+default_policy/sequential_1/q_hidden_0/biassave/Identity_55*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_56Identitysave/RestoreV2:56*
T0*
_output_shapes
:
И
save/AssignVariableOp_55AssignVariableOp-default_policy/sequential_1/q_hidden_0/kernelsave/Identity_56*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_57Identitysave/RestoreV2:57*
T0*
_output_shapes
:
Ж
save/AssignVariableOp_56AssignVariableOp+default_policy/sequential_1/q_hidden_1/biassave/Identity_57*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_58Identitysave/RestoreV2:58*
T0*
_output_shapes
:
И
save/AssignVariableOp_57AssignVariableOp-default_policy/sequential_1/q_hidden_1/kernelsave/Identity_58*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_59Identitysave/RestoreV2:59*
T0*
_output_shapes
:
Б
save/AssignVariableOp_58AssignVariableOp&default_policy/sequential_1/q_out/biassave/Identity_59*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_60Identitysave/RestoreV2:60*
T0*
_output_shapes
:
Г
save/AssignVariableOp_59AssignVariableOp(default_policy/sequential_1/q_out/kernelsave/Identity_60*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_61Identitysave/RestoreV2:61*
T0*
_output_shapes
:
Л
save/AssignVariableOp_60AssignVariableOp0default_policy/sequential_2/twin_q_hidden_0/biassave/Identity_61*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_62Identitysave/RestoreV2:62*
T0*
_output_shapes
:
Н
save/AssignVariableOp_61AssignVariableOp2default_policy/sequential_2/twin_q_hidden_0/kernelsave/Identity_62*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_63Identitysave/RestoreV2:63*
T0*
_output_shapes
:
Л
save/AssignVariableOp_62AssignVariableOp0default_policy/sequential_2/twin_q_hidden_1/biassave/Identity_63*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_64Identitysave/RestoreV2:64*
T0*
_output_shapes
:
Н
save/AssignVariableOp_63AssignVariableOp2default_policy/sequential_2/twin_q_hidden_1/kernelsave/Identity_64*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_65Identitysave/RestoreV2:65*
T0*
_output_shapes
:
Ж
save/AssignVariableOp_64AssignVariableOp+default_policy/sequential_2/twin_q_out/biassave/Identity_65*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_66Identitysave/RestoreV2:66*
T0*
_output_shapes
:
И
save/AssignVariableOp_65AssignVariableOp-default_policy/sequential_2/twin_q_out/kernelsave/Identity_66*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_67Identitysave/RestoreV2:67*
T0*
_output_shapes
:
Д
save/AssignVariableOp_66AssignVariableOp)default_policy/sequential_3/action_1/biassave/Identity_67*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_68Identitysave/RestoreV2:68*
T0*
_output_shapes
:
Ж
save/AssignVariableOp_67AssignVariableOp+default_policy/sequential_3/action_1/kernelsave/Identity_68*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_69Identitysave/RestoreV2:69*
T0*
_output_shapes
:
Д
save/AssignVariableOp_68AssignVariableOp)default_policy/sequential_3/action_2/biassave/Identity_69*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_70Identitysave/RestoreV2:70*
T0*
_output_shapes
:
Ж
save/AssignVariableOp_69AssignVariableOp+default_policy/sequential_3/action_2/kernelsave/Identity_70*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_71Identitysave/RestoreV2:71*
T0*
_output_shapes
:
Ж
save/AssignVariableOp_70AssignVariableOp+default_policy/sequential_3/action_out/biassave/Identity_71*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_72Identitysave/RestoreV2:72*
T0*
_output_shapes
:
И
save/AssignVariableOp_71AssignVariableOp-default_policy/sequential_3/action_out/kernelsave/Identity_72*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_73Identitysave/RestoreV2:73*
T0*
_output_shapes
:
Ж
save/AssignVariableOp_72AssignVariableOp+default_policy/sequential_4/q_hidden_0/biassave/Identity_73*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_74Identitysave/RestoreV2:74*
T0*
_output_shapes
:
И
save/AssignVariableOp_73AssignVariableOp-default_policy/sequential_4/q_hidden_0/kernelsave/Identity_74*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_75Identitysave/RestoreV2:75*
T0*
_output_shapes
:
Ж
save/AssignVariableOp_74AssignVariableOp+default_policy/sequential_4/q_hidden_1/biassave/Identity_75*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_76Identitysave/RestoreV2:76*
T0*
_output_shapes
:
И
save/AssignVariableOp_75AssignVariableOp-default_policy/sequential_4/q_hidden_1/kernelsave/Identity_76*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_77Identitysave/RestoreV2:77*
T0*
_output_shapes
:
Б
save/AssignVariableOp_76AssignVariableOp&default_policy/sequential_4/q_out/biassave/Identity_77*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_78Identitysave/RestoreV2:78*
T0*
_output_shapes
:
Г
save/AssignVariableOp_77AssignVariableOp(default_policy/sequential_4/q_out/kernelsave/Identity_78*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_79Identitysave/RestoreV2:79*
T0*
_output_shapes
:
Л
save/AssignVariableOp_78AssignVariableOp0default_policy/sequential_5/twin_q_hidden_0/biassave/Identity_79*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_80Identitysave/RestoreV2:80*
T0*
_output_shapes
:
Н
save/AssignVariableOp_79AssignVariableOp2default_policy/sequential_5/twin_q_hidden_0/kernelsave/Identity_80*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_81Identitysave/RestoreV2:81*
T0*
_output_shapes
:
Л
save/AssignVariableOp_80AssignVariableOp0default_policy/sequential_5/twin_q_hidden_1/biassave/Identity_81*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_82Identitysave/RestoreV2:82*
T0*
_output_shapes
:
Н
save/AssignVariableOp_81AssignVariableOp2default_policy/sequential_5/twin_q_hidden_1/kernelsave/Identity_82*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_83Identitysave/RestoreV2:83*
T0*
_output_shapes
:
Ж
save/AssignVariableOp_82AssignVariableOp+default_policy/sequential_5/twin_q_out/biassave/Identity_83*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_84Identitysave/RestoreV2:84*
T0*
_output_shapes
:
И
save/AssignVariableOp_83AssignVariableOp-default_policy/sequential_5/twin_q_out/kernelsave/Identity_84*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_85Identitysave/RestoreV2:85*
T0*
_output_shapes
:
Ј
save/AssignVariableOp_84AssignVariableOpdefault_policy/value_out/biassave/Identity_85*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_86Identitysave/RestoreV2:86*
T0*
_output_shapes
:
Њ
save/AssignVariableOp_85AssignVariableOpdefault_policy/value_out/kernelsave/Identity_86*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_87Identitysave/RestoreV2:87*
T0*
_output_shapes
:
Њ
save/AssignVariableOp_86AssignVariableOpdefault_policy/value_out_1/biassave/Identity_87*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
R
save/Identity_88Identitysave/RestoreV2:88*
T0*
_output_shapes
:
Ќ
save/AssignVariableOp_87AssignVariableOp!default_policy/value_out_1/kernelsave/Identity_88*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 

save/restore_shardNoOp^save/Assign^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_14^save/AssignVariableOp_15^save/AssignVariableOp_16^save/AssignVariableOp_17^save/AssignVariableOp_18^save/AssignVariableOp_19^save/AssignVariableOp_2^save/AssignVariableOp_20^save/AssignVariableOp_21^save/AssignVariableOp_22^save/AssignVariableOp_23^save/AssignVariableOp_24^save/AssignVariableOp_25^save/AssignVariableOp_26^save/AssignVariableOp_27^save/AssignVariableOp_28^save/AssignVariableOp_29^save/AssignVariableOp_3^save/AssignVariableOp_30^save/AssignVariableOp_31^save/AssignVariableOp_32^save/AssignVariableOp_33^save/AssignVariableOp_34^save/AssignVariableOp_35^save/AssignVariableOp_36^save/AssignVariableOp_37^save/AssignVariableOp_38^save/AssignVariableOp_39^save/AssignVariableOp_4^save/AssignVariableOp_40^save/AssignVariableOp_41^save/AssignVariableOp_42^save/AssignVariableOp_43^save/AssignVariableOp_44^save/AssignVariableOp_45^save/AssignVariableOp_46^save/AssignVariableOp_47^save/AssignVariableOp_48^save/AssignVariableOp_49^save/AssignVariableOp_5^save/AssignVariableOp_50^save/AssignVariableOp_51^save/AssignVariableOp_52^save/AssignVariableOp_53^save/AssignVariableOp_54^save/AssignVariableOp_55^save/AssignVariableOp_56^save/AssignVariableOp_57^save/AssignVariableOp_58^save/AssignVariableOp_59^save/AssignVariableOp_6^save/AssignVariableOp_60^save/AssignVariableOp_61^save/AssignVariableOp_62^save/AssignVariableOp_63^save/AssignVariableOp_64^save/AssignVariableOp_65^save/AssignVariableOp_66^save/AssignVariableOp_67^save/AssignVariableOp_68^save/AssignVariableOp_69^save/AssignVariableOp_7^save/AssignVariableOp_70^save/AssignVariableOp_71^save/AssignVariableOp_72^save/AssignVariableOp_73^save/AssignVariableOp_74^save/AssignVariableOp_75^save/AssignVariableOp_76^save/AssignVariableOp_77^save/AssignVariableOp_78^save/AssignVariableOp_79^save/AssignVariableOp_8^save/AssignVariableOp_80^save/AssignVariableOp_81^save/AssignVariableOp_82^save/AssignVariableOp_83^save/AssignVariableOp_84^save/AssignVariableOp_85^save/AssignVariableOp_86^save/AssignVariableOp_87^save/AssignVariableOp_9*&
 _has_manual_control_dependencies(
-
save/restore_allNoOp^save/restore_shard"
<
save/Const:0save/Identity:0save/restore_all (5 @F8"ы
cond_contextкз

default_policy/cond/cond_textdefault_policy/cond/pred_id:0default_policy/cond/switch_t:0 *Ѕ
 default_policy/clip_by_value_1:0
default_policy/cond/Switch_1:0
default_policy/cond/Switch_1:1
default_policy/cond/pred_id:0
default_policy/cond/switch_t:0>
default_policy/cond/pred_id:0default_policy/cond/pred_id:0B
 default_policy/clip_by_value_1:0default_policy/cond/Switch_1:1

default_policy/cond/cond_text_1default_policy/cond/pred_id:0default_policy/cond/switch_f:0*Ѕ
 default_policy/clip_by_value_4:0
default_policy/cond/Switch_2:0
default_policy/cond/Switch_2:1
default_policy/cond/pred_id:0
default_policy/cond/switch_f:0>
default_policy/cond/pred_id:0default_policy/cond/pred_id:0B
 default_policy/clip_by_value_4:0default_policy/cond/Switch_2:0

default_policy/cond_1/cond_textdefault_policy/cond_1/pred_id:0 default_policy/cond_1/switch_t:0 *
 default_policy/cond_1/Switch_1:0
 default_policy/cond_1/Switch_1:1
default_policy/cond_1/pred_id:0
 default_policy/cond_1/switch_t:0
default_policy/sub_3:0B
default_policy/cond_1/pred_id:0default_policy/cond_1/pred_id:0:
default_policy/sub_3:0 default_policy/cond_1/Switch_1:1
Г
!default_policy/cond_1/cond_text_1default_policy/cond_1/pred_id:0 default_policy/cond_1/switch_f:0*Ъ
default_policy/cond/Merge:0
$default_policy/cond_1/Shape/Switch:0
default_policy/cond_1/Shape:0
default_policy/cond_1/pred_id:0
+default_policy/cond_1/strided_slice/stack:0
-default_policy/cond_1/strided_slice/stack_1:0
-default_policy/cond_1/strided_slice/stack_2:0
%default_policy/cond_1/strided_slice:0
 default_policy/cond_1/switch_f:0
#default_policy/cond_1/zeros/Const:0
$default_policy/cond_1/zeros/packed:0
default_policy/cond_1/zeros:0B
default_policy/cond_1/pred_id:0default_policy/cond_1/pred_id:0C
default_policy/cond/Merge:0$default_policy/cond_1/Shape/Switch:0"Ќ
global_step

default_policy/global_step:0!default_policy/global_step/Assign!default_policy/global_step/read:02.default_policy/global_step/Initializer/zeros:0H"h
train_op\
Z
default_policy/Adam
default_policy/Adam_1
default_policy/Adam_2
default_policy/Adam_3"љM
trainable_variablesсMоM
Л
!default_policy/value_out/kernel:0&default_policy/value_out/kernel/Assign5default_policy/value_out/kernel/Read/ReadVariableOp:0(23default_policy/value_out/kernel/Initializer/Const:08
Г
default_policy/value_out/bias:0$default_policy/value_out/bias/Assign3default_policy/value_out/bias/Read/ReadVariableOp:0(21default_policy/value_out/bias/Initializer/zeros:08
ь
+default_policy/sequential/action_1/kernel:00default_policy/sequential/action_1/kernel/Assign?default_policy/sequential/action_1/kernel/Read/ReadVariableOp:0(2Fdefault_policy/sequential/action_1/kernel/Initializer/random_uniform:08
л
)default_policy/sequential/action_1/bias:0.default_policy/sequential/action_1/bias/Assign=default_policy/sequential/action_1/bias/Read/ReadVariableOp:0(2;default_policy/sequential/action_1/bias/Initializer/zeros:08
ь
+default_policy/sequential/action_2/kernel:00default_policy/sequential/action_2/kernel/Assign?default_policy/sequential/action_2/kernel/Read/ReadVariableOp:0(2Fdefault_policy/sequential/action_2/kernel/Initializer/random_uniform:08
л
)default_policy/sequential/action_2/bias:0.default_policy/sequential/action_2/bias/Assign=default_policy/sequential/action_2/bias/Read/ReadVariableOp:0(2;default_policy/sequential/action_2/bias/Initializer/zeros:08
є
-default_policy/sequential/action_out/kernel:02default_policy/sequential/action_out/kernel/AssignAdefault_policy/sequential/action_out/kernel/Read/ReadVariableOp:0(2Hdefault_policy/sequential/action_out/kernel/Initializer/random_uniform:08
у
+default_policy/sequential/action_out/bias:00default_policy/sequential/action_out/bias/Assign?default_policy/sequential/action_out/bias/Read/ReadVariableOp:0(2=default_policy/sequential/action_out/bias/Initializer/zeros:08
ќ
/default_policy/sequential_1/q_hidden_0/kernel:04default_policy/sequential_1/q_hidden_0/kernel/AssignCdefault_policy/sequential_1/q_hidden_0/kernel/Read/ReadVariableOp:0(2Jdefault_policy/sequential_1/q_hidden_0/kernel/Initializer/random_uniform:08
ы
-default_policy/sequential_1/q_hidden_0/bias:02default_policy/sequential_1/q_hidden_0/bias/AssignAdefault_policy/sequential_1/q_hidden_0/bias/Read/ReadVariableOp:0(2?default_policy/sequential_1/q_hidden_0/bias/Initializer/zeros:08
ќ
/default_policy/sequential_1/q_hidden_1/kernel:04default_policy/sequential_1/q_hidden_1/kernel/AssignCdefault_policy/sequential_1/q_hidden_1/kernel/Read/ReadVariableOp:0(2Jdefault_policy/sequential_1/q_hidden_1/kernel/Initializer/random_uniform:08
ы
-default_policy/sequential_1/q_hidden_1/bias:02default_policy/sequential_1/q_hidden_1/bias/AssignAdefault_policy/sequential_1/q_hidden_1/bias/Read/ReadVariableOp:0(2?default_policy/sequential_1/q_hidden_1/bias/Initializer/zeros:08
ш
*default_policy/sequential_1/q_out/kernel:0/default_policy/sequential_1/q_out/kernel/Assign>default_policy/sequential_1/q_out/kernel/Read/ReadVariableOp:0(2Edefault_policy/sequential_1/q_out/kernel/Initializer/random_uniform:08
з
(default_policy/sequential_1/q_out/bias:0-default_policy/sequential_1/q_out/bias/Assign<default_policy/sequential_1/q_out/bias/Read/ReadVariableOp:0(2:default_policy/sequential_1/q_out/bias/Initializer/zeros:08

4default_policy/sequential_2/twin_q_hidden_0/kernel:09default_policy/sequential_2/twin_q_hidden_0/kernel/AssignHdefault_policy/sequential_2/twin_q_hidden_0/kernel/Read/ReadVariableOp:0(2Odefault_policy/sequential_2/twin_q_hidden_0/kernel/Initializer/random_uniform:08
џ
2default_policy/sequential_2/twin_q_hidden_0/bias:07default_policy/sequential_2/twin_q_hidden_0/bias/AssignFdefault_policy/sequential_2/twin_q_hidden_0/bias/Read/ReadVariableOp:0(2Ddefault_policy/sequential_2/twin_q_hidden_0/bias/Initializer/zeros:08

4default_policy/sequential_2/twin_q_hidden_1/kernel:09default_policy/sequential_2/twin_q_hidden_1/kernel/AssignHdefault_policy/sequential_2/twin_q_hidden_1/kernel/Read/ReadVariableOp:0(2Odefault_policy/sequential_2/twin_q_hidden_1/kernel/Initializer/random_uniform:08
џ
2default_policy/sequential_2/twin_q_hidden_1/bias:07default_policy/sequential_2/twin_q_hidden_1/bias/AssignFdefault_policy/sequential_2/twin_q_hidden_1/bias/Read/ReadVariableOp:0(2Ddefault_policy/sequential_2/twin_q_hidden_1/bias/Initializer/zeros:08
ќ
/default_policy/sequential_2/twin_q_out/kernel:04default_policy/sequential_2/twin_q_out/kernel/AssignCdefault_policy/sequential_2/twin_q_out/kernel/Read/ReadVariableOp:0(2Jdefault_policy/sequential_2/twin_q_out/kernel/Initializer/random_uniform:08
ы
-default_policy/sequential_2/twin_q_out/bias:02default_policy/sequential_2/twin_q_out/bias/AssignAdefault_policy/sequential_2/twin_q_out/bias/Read/ReadVariableOp:0(2?default_policy/sequential_2/twin_q_out/bias/Initializer/zeros:08
Ї
default_policy/log_alpha:0default_policy/log_alpha/Assign.default_policy/log_alpha/Read/ReadVariableOp:0(24default_policy/log_alpha/Initializer/initial_value:08
У
#default_policy/value_out_1/kernel:0(default_policy/value_out_1/kernel/Assign7default_policy/value_out_1/kernel/Read/ReadVariableOp:0(25default_policy/value_out_1/kernel/Initializer/Const:08
Л
!default_policy/value_out_1/bias:0&default_policy/value_out_1/bias/Assign5default_policy/value_out_1/bias/Read/ReadVariableOp:0(23default_policy/value_out_1/bias/Initializer/zeros:08
є
-default_policy/sequential_3/action_1/kernel:02default_policy/sequential_3/action_1/kernel/AssignAdefault_policy/sequential_3/action_1/kernel/Read/ReadVariableOp:0(2Hdefault_policy/sequential_3/action_1/kernel/Initializer/random_uniform:08
у
+default_policy/sequential_3/action_1/bias:00default_policy/sequential_3/action_1/bias/Assign?default_policy/sequential_3/action_1/bias/Read/ReadVariableOp:0(2=default_policy/sequential_3/action_1/bias/Initializer/zeros:08
є
-default_policy/sequential_3/action_2/kernel:02default_policy/sequential_3/action_2/kernel/AssignAdefault_policy/sequential_3/action_2/kernel/Read/ReadVariableOp:0(2Hdefault_policy/sequential_3/action_2/kernel/Initializer/random_uniform:08
у
+default_policy/sequential_3/action_2/bias:00default_policy/sequential_3/action_2/bias/Assign?default_policy/sequential_3/action_2/bias/Read/ReadVariableOp:0(2=default_policy/sequential_3/action_2/bias/Initializer/zeros:08
ќ
/default_policy/sequential_3/action_out/kernel:04default_policy/sequential_3/action_out/kernel/AssignCdefault_policy/sequential_3/action_out/kernel/Read/ReadVariableOp:0(2Jdefault_policy/sequential_3/action_out/kernel/Initializer/random_uniform:08
ы
-default_policy/sequential_3/action_out/bias:02default_policy/sequential_3/action_out/bias/AssignAdefault_policy/sequential_3/action_out/bias/Read/ReadVariableOp:0(2?default_policy/sequential_3/action_out/bias/Initializer/zeros:08
ќ
/default_policy/sequential_4/q_hidden_0/kernel:04default_policy/sequential_4/q_hidden_0/kernel/AssignCdefault_policy/sequential_4/q_hidden_0/kernel/Read/ReadVariableOp:0(2Jdefault_policy/sequential_4/q_hidden_0/kernel/Initializer/random_uniform:08
ы
-default_policy/sequential_4/q_hidden_0/bias:02default_policy/sequential_4/q_hidden_0/bias/AssignAdefault_policy/sequential_4/q_hidden_0/bias/Read/ReadVariableOp:0(2?default_policy/sequential_4/q_hidden_0/bias/Initializer/zeros:08
ќ
/default_policy/sequential_4/q_hidden_1/kernel:04default_policy/sequential_4/q_hidden_1/kernel/AssignCdefault_policy/sequential_4/q_hidden_1/kernel/Read/ReadVariableOp:0(2Jdefault_policy/sequential_4/q_hidden_1/kernel/Initializer/random_uniform:08
ы
-default_policy/sequential_4/q_hidden_1/bias:02default_policy/sequential_4/q_hidden_1/bias/AssignAdefault_policy/sequential_4/q_hidden_1/bias/Read/ReadVariableOp:0(2?default_policy/sequential_4/q_hidden_1/bias/Initializer/zeros:08
ш
*default_policy/sequential_4/q_out/kernel:0/default_policy/sequential_4/q_out/kernel/Assign>default_policy/sequential_4/q_out/kernel/Read/ReadVariableOp:0(2Edefault_policy/sequential_4/q_out/kernel/Initializer/random_uniform:08
з
(default_policy/sequential_4/q_out/bias:0-default_policy/sequential_4/q_out/bias/Assign<default_policy/sequential_4/q_out/bias/Read/ReadVariableOp:0(2:default_policy/sequential_4/q_out/bias/Initializer/zeros:08

4default_policy/sequential_5/twin_q_hidden_0/kernel:09default_policy/sequential_5/twin_q_hidden_0/kernel/AssignHdefault_policy/sequential_5/twin_q_hidden_0/kernel/Read/ReadVariableOp:0(2Odefault_policy/sequential_5/twin_q_hidden_0/kernel/Initializer/random_uniform:08
џ
2default_policy/sequential_5/twin_q_hidden_0/bias:07default_policy/sequential_5/twin_q_hidden_0/bias/AssignFdefault_policy/sequential_5/twin_q_hidden_0/bias/Read/ReadVariableOp:0(2Ddefault_policy/sequential_5/twin_q_hidden_0/bias/Initializer/zeros:08

4default_policy/sequential_5/twin_q_hidden_1/kernel:09default_policy/sequential_5/twin_q_hidden_1/kernel/AssignHdefault_policy/sequential_5/twin_q_hidden_1/kernel/Read/ReadVariableOp:0(2Odefault_policy/sequential_5/twin_q_hidden_1/kernel/Initializer/random_uniform:08
џ
2default_policy/sequential_5/twin_q_hidden_1/bias:07default_policy/sequential_5/twin_q_hidden_1/bias/AssignFdefault_policy/sequential_5/twin_q_hidden_1/bias/Read/ReadVariableOp:0(2Ddefault_policy/sequential_5/twin_q_hidden_1/bias/Initializer/zeros:08
ќ
/default_policy/sequential_5/twin_q_out/kernel:04default_policy/sequential_5/twin_q_out/kernel/AssignCdefault_policy/sequential_5/twin_q_out/kernel/Read/ReadVariableOp:0(2Jdefault_policy/sequential_5/twin_q_out/kernel/Initializer/random_uniform:08
ы
-default_policy/sequential_5/twin_q_out/bias:02default_policy/sequential_5/twin_q_out/bias/AssignAdefault_policy/sequential_5/twin_q_out/bias/Read/ReadVariableOp:0(2?default_policy/sequential_5/twin_q_out/bias/Initializer/zeros:08
Џ
default_policy/log_alpha_1:0!default_policy/log_alpha_1/Assign0default_policy/log_alpha_1/Read/ReadVariableOp:0(26default_policy/log_alpha_1/Initializer/initial_value:08"тИ
	variablesгИЯИ

default_policy/global_step:0!default_policy/global_step/Assign!default_policy/global_step/read:02.default_policy/global_step/Initializer/zeros:0H
Л
!default_policy/value_out/kernel:0&default_policy/value_out/kernel/Assign5default_policy/value_out/kernel/Read/ReadVariableOp:0(23default_policy/value_out/kernel/Initializer/Const:08
Г
default_policy/value_out/bias:0$default_policy/value_out/bias/Assign3default_policy/value_out/bias/Read/ReadVariableOp:0(21default_policy/value_out/bias/Initializer/zeros:08
ь
+default_policy/sequential/action_1/kernel:00default_policy/sequential/action_1/kernel/Assign?default_policy/sequential/action_1/kernel/Read/ReadVariableOp:0(2Fdefault_policy/sequential/action_1/kernel/Initializer/random_uniform:08
л
)default_policy/sequential/action_1/bias:0.default_policy/sequential/action_1/bias/Assign=default_policy/sequential/action_1/bias/Read/ReadVariableOp:0(2;default_policy/sequential/action_1/bias/Initializer/zeros:08
ь
+default_policy/sequential/action_2/kernel:00default_policy/sequential/action_2/kernel/Assign?default_policy/sequential/action_2/kernel/Read/ReadVariableOp:0(2Fdefault_policy/sequential/action_2/kernel/Initializer/random_uniform:08
л
)default_policy/sequential/action_2/bias:0.default_policy/sequential/action_2/bias/Assign=default_policy/sequential/action_2/bias/Read/ReadVariableOp:0(2;default_policy/sequential/action_2/bias/Initializer/zeros:08
є
-default_policy/sequential/action_out/kernel:02default_policy/sequential/action_out/kernel/AssignAdefault_policy/sequential/action_out/kernel/Read/ReadVariableOp:0(2Hdefault_policy/sequential/action_out/kernel/Initializer/random_uniform:08
у
+default_policy/sequential/action_out/bias:00default_policy/sequential/action_out/bias/Assign?default_policy/sequential/action_out/bias/Read/ReadVariableOp:0(2=default_policy/sequential/action_out/bias/Initializer/zeros:08
ќ
/default_policy/sequential_1/q_hidden_0/kernel:04default_policy/sequential_1/q_hidden_0/kernel/AssignCdefault_policy/sequential_1/q_hidden_0/kernel/Read/ReadVariableOp:0(2Jdefault_policy/sequential_1/q_hidden_0/kernel/Initializer/random_uniform:08
ы
-default_policy/sequential_1/q_hidden_0/bias:02default_policy/sequential_1/q_hidden_0/bias/AssignAdefault_policy/sequential_1/q_hidden_0/bias/Read/ReadVariableOp:0(2?default_policy/sequential_1/q_hidden_0/bias/Initializer/zeros:08
ќ
/default_policy/sequential_1/q_hidden_1/kernel:04default_policy/sequential_1/q_hidden_1/kernel/AssignCdefault_policy/sequential_1/q_hidden_1/kernel/Read/ReadVariableOp:0(2Jdefault_policy/sequential_1/q_hidden_1/kernel/Initializer/random_uniform:08
ы
-default_policy/sequential_1/q_hidden_1/bias:02default_policy/sequential_1/q_hidden_1/bias/AssignAdefault_policy/sequential_1/q_hidden_1/bias/Read/ReadVariableOp:0(2?default_policy/sequential_1/q_hidden_1/bias/Initializer/zeros:08
ш
*default_policy/sequential_1/q_out/kernel:0/default_policy/sequential_1/q_out/kernel/Assign>default_policy/sequential_1/q_out/kernel/Read/ReadVariableOp:0(2Edefault_policy/sequential_1/q_out/kernel/Initializer/random_uniform:08
з
(default_policy/sequential_1/q_out/bias:0-default_policy/sequential_1/q_out/bias/Assign<default_policy/sequential_1/q_out/bias/Read/ReadVariableOp:0(2:default_policy/sequential_1/q_out/bias/Initializer/zeros:08

4default_policy/sequential_2/twin_q_hidden_0/kernel:09default_policy/sequential_2/twin_q_hidden_0/kernel/AssignHdefault_policy/sequential_2/twin_q_hidden_0/kernel/Read/ReadVariableOp:0(2Odefault_policy/sequential_2/twin_q_hidden_0/kernel/Initializer/random_uniform:08
џ
2default_policy/sequential_2/twin_q_hidden_0/bias:07default_policy/sequential_2/twin_q_hidden_0/bias/AssignFdefault_policy/sequential_2/twin_q_hidden_0/bias/Read/ReadVariableOp:0(2Ddefault_policy/sequential_2/twin_q_hidden_0/bias/Initializer/zeros:08

4default_policy/sequential_2/twin_q_hidden_1/kernel:09default_policy/sequential_2/twin_q_hidden_1/kernel/AssignHdefault_policy/sequential_2/twin_q_hidden_1/kernel/Read/ReadVariableOp:0(2Odefault_policy/sequential_2/twin_q_hidden_1/kernel/Initializer/random_uniform:08
џ
2default_policy/sequential_2/twin_q_hidden_1/bias:07default_policy/sequential_2/twin_q_hidden_1/bias/AssignFdefault_policy/sequential_2/twin_q_hidden_1/bias/Read/ReadVariableOp:0(2Ddefault_policy/sequential_2/twin_q_hidden_1/bias/Initializer/zeros:08
ќ
/default_policy/sequential_2/twin_q_out/kernel:04default_policy/sequential_2/twin_q_out/kernel/AssignCdefault_policy/sequential_2/twin_q_out/kernel/Read/ReadVariableOp:0(2Jdefault_policy/sequential_2/twin_q_out/kernel/Initializer/random_uniform:08
ы
-default_policy/sequential_2/twin_q_out/bias:02default_policy/sequential_2/twin_q_out/bias/AssignAdefault_policy/sequential_2/twin_q_out/bias/Read/ReadVariableOp:0(2?default_policy/sequential_2/twin_q_out/bias/Initializer/zeros:08
Ї
default_policy/log_alpha:0default_policy/log_alpha/Assign.default_policy/log_alpha/Read/ReadVariableOp:0(24default_policy/log_alpha/Initializer/initial_value:08
У
#default_policy/value_out_1/kernel:0(default_policy/value_out_1/kernel/Assign7default_policy/value_out_1/kernel/Read/ReadVariableOp:0(25default_policy/value_out_1/kernel/Initializer/Const:08
Л
!default_policy/value_out_1/bias:0&default_policy/value_out_1/bias/Assign5default_policy/value_out_1/bias/Read/ReadVariableOp:0(23default_policy/value_out_1/bias/Initializer/zeros:08
є
-default_policy/sequential_3/action_1/kernel:02default_policy/sequential_3/action_1/kernel/AssignAdefault_policy/sequential_3/action_1/kernel/Read/ReadVariableOp:0(2Hdefault_policy/sequential_3/action_1/kernel/Initializer/random_uniform:08
у
+default_policy/sequential_3/action_1/bias:00default_policy/sequential_3/action_1/bias/Assign?default_policy/sequential_3/action_1/bias/Read/ReadVariableOp:0(2=default_policy/sequential_3/action_1/bias/Initializer/zeros:08
є
-default_policy/sequential_3/action_2/kernel:02default_policy/sequential_3/action_2/kernel/AssignAdefault_policy/sequential_3/action_2/kernel/Read/ReadVariableOp:0(2Hdefault_policy/sequential_3/action_2/kernel/Initializer/random_uniform:08
у
+default_policy/sequential_3/action_2/bias:00default_policy/sequential_3/action_2/bias/Assign?default_policy/sequential_3/action_2/bias/Read/ReadVariableOp:0(2=default_policy/sequential_3/action_2/bias/Initializer/zeros:08
ќ
/default_policy/sequential_3/action_out/kernel:04default_policy/sequential_3/action_out/kernel/AssignCdefault_policy/sequential_3/action_out/kernel/Read/ReadVariableOp:0(2Jdefault_policy/sequential_3/action_out/kernel/Initializer/random_uniform:08
ы
-default_policy/sequential_3/action_out/bias:02default_policy/sequential_3/action_out/bias/AssignAdefault_policy/sequential_3/action_out/bias/Read/ReadVariableOp:0(2?default_policy/sequential_3/action_out/bias/Initializer/zeros:08
ќ
/default_policy/sequential_4/q_hidden_0/kernel:04default_policy/sequential_4/q_hidden_0/kernel/AssignCdefault_policy/sequential_4/q_hidden_0/kernel/Read/ReadVariableOp:0(2Jdefault_policy/sequential_4/q_hidden_0/kernel/Initializer/random_uniform:08
ы
-default_policy/sequential_4/q_hidden_0/bias:02default_policy/sequential_4/q_hidden_0/bias/AssignAdefault_policy/sequential_4/q_hidden_0/bias/Read/ReadVariableOp:0(2?default_policy/sequential_4/q_hidden_0/bias/Initializer/zeros:08
ќ
/default_policy/sequential_4/q_hidden_1/kernel:04default_policy/sequential_4/q_hidden_1/kernel/AssignCdefault_policy/sequential_4/q_hidden_1/kernel/Read/ReadVariableOp:0(2Jdefault_policy/sequential_4/q_hidden_1/kernel/Initializer/random_uniform:08
ы
-default_policy/sequential_4/q_hidden_1/bias:02default_policy/sequential_4/q_hidden_1/bias/AssignAdefault_policy/sequential_4/q_hidden_1/bias/Read/ReadVariableOp:0(2?default_policy/sequential_4/q_hidden_1/bias/Initializer/zeros:08
ш
*default_policy/sequential_4/q_out/kernel:0/default_policy/sequential_4/q_out/kernel/Assign>default_policy/sequential_4/q_out/kernel/Read/ReadVariableOp:0(2Edefault_policy/sequential_4/q_out/kernel/Initializer/random_uniform:08
з
(default_policy/sequential_4/q_out/bias:0-default_policy/sequential_4/q_out/bias/Assign<default_policy/sequential_4/q_out/bias/Read/ReadVariableOp:0(2:default_policy/sequential_4/q_out/bias/Initializer/zeros:08

4default_policy/sequential_5/twin_q_hidden_0/kernel:09default_policy/sequential_5/twin_q_hidden_0/kernel/AssignHdefault_policy/sequential_5/twin_q_hidden_0/kernel/Read/ReadVariableOp:0(2Odefault_policy/sequential_5/twin_q_hidden_0/kernel/Initializer/random_uniform:08
џ
2default_policy/sequential_5/twin_q_hidden_0/bias:07default_policy/sequential_5/twin_q_hidden_0/bias/AssignFdefault_policy/sequential_5/twin_q_hidden_0/bias/Read/ReadVariableOp:0(2Ddefault_policy/sequential_5/twin_q_hidden_0/bias/Initializer/zeros:08

4default_policy/sequential_5/twin_q_hidden_1/kernel:09default_policy/sequential_5/twin_q_hidden_1/kernel/AssignHdefault_policy/sequential_5/twin_q_hidden_1/kernel/Read/ReadVariableOp:0(2Odefault_policy/sequential_5/twin_q_hidden_1/kernel/Initializer/random_uniform:08
џ
2default_policy/sequential_5/twin_q_hidden_1/bias:07default_policy/sequential_5/twin_q_hidden_1/bias/AssignFdefault_policy/sequential_5/twin_q_hidden_1/bias/Read/ReadVariableOp:0(2Ddefault_policy/sequential_5/twin_q_hidden_1/bias/Initializer/zeros:08
ќ
/default_policy/sequential_5/twin_q_out/kernel:04default_policy/sequential_5/twin_q_out/kernel/AssignCdefault_policy/sequential_5/twin_q_out/kernel/Read/ReadVariableOp:0(2Jdefault_policy/sequential_5/twin_q_out/kernel/Initializer/random_uniform:08
ы
-default_policy/sequential_5/twin_q_out/bias:02default_policy/sequential_5/twin_q_out/bias/AssignAdefault_policy/sequential_5/twin_q_out/bias/Read/ReadVariableOp:0(2?default_policy/sequential_5/twin_q_out/bias/Initializer/zeros:08
Џ
default_policy/log_alpha_1:0!default_policy/log_alpha_1/Assign0default_policy/log_alpha_1/Read/ReadVariableOp:0(26default_policy/log_alpha_1/Initializer/initial_value:08
­
default_policy/beta1_power:0!default_policy/beta1_power/Assign0default_policy/beta1_power/Read/ReadVariableOp:0(26default_policy/beta1_power/Initializer/initial_value:0
­
default_policy/beta2_power:0!default_policy/beta2_power/Assign0default_policy/beta2_power/Read/ReadVariableOp:0(26default_policy/beta2_power/Initializer/initial_value:0
Б
?default_policy/default_policy/sequential/action_1/kernel/Adam:0Ddefault_policy/default_policy/sequential/action_1/kernel/Adam/AssignSdefault_policy/default_policy/sequential/action_1/kernel/Adam/Read/ReadVariableOp:0(2Qdefault_policy/default_policy/sequential/action_1/kernel/Adam/Initializer/zeros:0
Й
Adefault_policy/default_policy/sequential/action_1/kernel/Adam_1:0Fdefault_policy/default_policy/sequential/action_1/kernel/Adam_1/AssignUdefault_policy/default_policy/sequential/action_1/kernel/Adam_1/Read/ReadVariableOp:0(2Sdefault_policy/default_policy/sequential/action_1/kernel/Adam_1/Initializer/zeros:0
Љ
=default_policy/default_policy/sequential/action_1/bias/Adam:0Bdefault_policy/default_policy/sequential/action_1/bias/Adam/AssignQdefault_policy/default_policy/sequential/action_1/bias/Adam/Read/ReadVariableOp:0(2Odefault_policy/default_policy/sequential/action_1/bias/Adam/Initializer/zeros:0
Б
?default_policy/default_policy/sequential/action_1/bias/Adam_1:0Ddefault_policy/default_policy/sequential/action_1/bias/Adam_1/AssignSdefault_policy/default_policy/sequential/action_1/bias/Adam_1/Read/ReadVariableOp:0(2Qdefault_policy/default_policy/sequential/action_1/bias/Adam_1/Initializer/zeros:0
Б
?default_policy/default_policy/sequential/action_2/kernel/Adam:0Ddefault_policy/default_policy/sequential/action_2/kernel/Adam/AssignSdefault_policy/default_policy/sequential/action_2/kernel/Adam/Read/ReadVariableOp:0(2Qdefault_policy/default_policy/sequential/action_2/kernel/Adam/Initializer/zeros:0
Й
Adefault_policy/default_policy/sequential/action_2/kernel/Adam_1:0Fdefault_policy/default_policy/sequential/action_2/kernel/Adam_1/AssignUdefault_policy/default_policy/sequential/action_2/kernel/Adam_1/Read/ReadVariableOp:0(2Sdefault_policy/default_policy/sequential/action_2/kernel/Adam_1/Initializer/zeros:0
Љ
=default_policy/default_policy/sequential/action_2/bias/Adam:0Bdefault_policy/default_policy/sequential/action_2/bias/Adam/AssignQdefault_policy/default_policy/sequential/action_2/bias/Adam/Read/ReadVariableOp:0(2Odefault_policy/default_policy/sequential/action_2/bias/Adam/Initializer/zeros:0
Б
?default_policy/default_policy/sequential/action_2/bias/Adam_1:0Ddefault_policy/default_policy/sequential/action_2/bias/Adam_1/AssignSdefault_policy/default_policy/sequential/action_2/bias/Adam_1/Read/ReadVariableOp:0(2Qdefault_policy/default_policy/sequential/action_2/bias/Adam_1/Initializer/zeros:0
Й
Adefault_policy/default_policy/sequential/action_out/kernel/Adam:0Fdefault_policy/default_policy/sequential/action_out/kernel/Adam/AssignUdefault_policy/default_policy/sequential/action_out/kernel/Adam/Read/ReadVariableOp:0(2Sdefault_policy/default_policy/sequential/action_out/kernel/Adam/Initializer/zeros:0
С
Cdefault_policy/default_policy/sequential/action_out/kernel/Adam_1:0Hdefault_policy/default_policy/sequential/action_out/kernel/Adam_1/AssignWdefault_policy/default_policy/sequential/action_out/kernel/Adam_1/Read/ReadVariableOp:0(2Udefault_policy/default_policy/sequential/action_out/kernel/Adam_1/Initializer/zeros:0
Б
?default_policy/default_policy/sequential/action_out/bias/Adam:0Ddefault_policy/default_policy/sequential/action_out/bias/Adam/AssignSdefault_policy/default_policy/sequential/action_out/bias/Adam/Read/ReadVariableOp:0(2Qdefault_policy/default_policy/sequential/action_out/bias/Adam/Initializer/zeros:0
Й
Adefault_policy/default_policy/sequential/action_out/bias/Adam_1:0Fdefault_policy/default_policy/sequential/action_out/bias/Adam_1/AssignUdefault_policy/default_policy/sequential/action_out/bias/Adam_1/Read/ReadVariableOp:0(2Sdefault_policy/default_policy/sequential/action_out/bias/Adam_1/Initializer/zeros:0
Е
default_policy/beta1_power_1:0#default_policy/beta1_power_1/Assign2default_policy/beta1_power_1/Read/ReadVariableOp:0(28default_policy/beta1_power_1/Initializer/initial_value:0
Е
default_policy/beta2_power_1:0#default_policy/beta2_power_1/Assign2default_policy/beta2_power_1/Read/ReadVariableOp:0(28default_policy/beta2_power_1/Initializer/initial_value:0
С
Cdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam:0Hdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam/AssignWdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam/Read/ReadVariableOp:0(2Udefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam/Initializer/zeros:0
Щ
Edefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam_1:0Jdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam_1/AssignYdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam_1/Read/ReadVariableOp:0(2Wdefault_policy/default_policy/sequential_1/q_hidden_0/kernel/Adam_1/Initializer/zeros:0
Й
Adefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam:0Fdefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam/AssignUdefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam/Read/ReadVariableOp:0(2Sdefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam/Initializer/zeros:0
С
Cdefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam_1:0Hdefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam_1/AssignWdefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam_1/Read/ReadVariableOp:0(2Udefault_policy/default_policy/sequential_1/q_hidden_0/bias/Adam_1/Initializer/zeros:0
С
Cdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam:0Hdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam/AssignWdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam/Read/ReadVariableOp:0(2Udefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam/Initializer/zeros:0
Щ
Edefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam_1:0Jdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam_1/AssignYdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam_1/Read/ReadVariableOp:0(2Wdefault_policy/default_policy/sequential_1/q_hidden_1/kernel/Adam_1/Initializer/zeros:0
Й
Adefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam:0Fdefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam/AssignUdefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam/Read/ReadVariableOp:0(2Sdefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam/Initializer/zeros:0
С
Cdefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam_1:0Hdefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam_1/AssignWdefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam_1/Read/ReadVariableOp:0(2Udefault_policy/default_policy/sequential_1/q_hidden_1/bias/Adam_1/Initializer/zeros:0
­
>default_policy/default_policy/sequential_1/q_out/kernel/Adam:0Cdefault_policy/default_policy/sequential_1/q_out/kernel/Adam/AssignRdefault_policy/default_policy/sequential_1/q_out/kernel/Adam/Read/ReadVariableOp:0(2Pdefault_policy/default_policy/sequential_1/q_out/kernel/Adam/Initializer/zeros:0
Е
@default_policy/default_policy/sequential_1/q_out/kernel/Adam_1:0Edefault_policy/default_policy/sequential_1/q_out/kernel/Adam_1/AssignTdefault_policy/default_policy/sequential_1/q_out/kernel/Adam_1/Read/ReadVariableOp:0(2Rdefault_policy/default_policy/sequential_1/q_out/kernel/Adam_1/Initializer/zeros:0
Ѕ
<default_policy/default_policy/sequential_1/q_out/bias/Adam:0Adefault_policy/default_policy/sequential_1/q_out/bias/Adam/AssignPdefault_policy/default_policy/sequential_1/q_out/bias/Adam/Read/ReadVariableOp:0(2Ndefault_policy/default_policy/sequential_1/q_out/bias/Adam/Initializer/zeros:0
­
>default_policy/default_policy/sequential_1/q_out/bias/Adam_1:0Cdefault_policy/default_policy/sequential_1/q_out/bias/Adam_1/AssignRdefault_policy/default_policy/sequential_1/q_out/bias/Adam_1/Read/ReadVariableOp:0(2Pdefault_policy/default_policy/sequential_1/q_out/bias/Adam_1/Initializer/zeros:0
Е
default_policy/beta1_power_2:0#default_policy/beta1_power_2/Assign2default_policy/beta1_power_2/Read/ReadVariableOp:0(28default_policy/beta1_power_2/Initializer/initial_value:0
Е
default_policy/beta2_power_2:0#default_policy/beta2_power_2/Assign2default_policy/beta2_power_2/Read/ReadVariableOp:0(28default_policy/beta2_power_2/Initializer/initial_value:0
е
Hdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam:0Mdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam/Assign\default_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam/Read/ReadVariableOp:0(2Zdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam/Initializer/zeros:0
н
Jdefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam_1:0Odefault_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam_1/Assign^default_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam_1/Read/ReadVariableOp:0(2\default_policy/default_policy/sequential_2/twin_q_hidden_0/kernel/Adam_1/Initializer/zeros:0
Э
Fdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam:0Kdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam/AssignZdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam/Read/ReadVariableOp:0(2Xdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam/Initializer/zeros:0
е
Hdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam_1:0Mdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam_1/Assign\default_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam_1/Read/ReadVariableOp:0(2Zdefault_policy/default_policy/sequential_2/twin_q_hidden_0/bias/Adam_1/Initializer/zeros:0
е
Hdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam:0Mdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam/Assign\default_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam/Read/ReadVariableOp:0(2Zdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam/Initializer/zeros:0
н
Jdefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam_1:0Odefault_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam_1/Assign^default_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam_1/Read/ReadVariableOp:0(2\default_policy/default_policy/sequential_2/twin_q_hidden_1/kernel/Adam_1/Initializer/zeros:0
Э
Fdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam:0Kdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam/AssignZdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam/Read/ReadVariableOp:0(2Xdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam/Initializer/zeros:0
е
Hdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam_1:0Mdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam_1/Assign\default_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam_1/Read/ReadVariableOp:0(2Zdefault_policy/default_policy/sequential_2/twin_q_hidden_1/bias/Adam_1/Initializer/zeros:0
С
Cdefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam:0Hdefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam/AssignWdefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam/Read/ReadVariableOp:0(2Udefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam/Initializer/zeros:0
Щ
Edefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam_1:0Jdefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam_1/AssignYdefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam_1/Read/ReadVariableOp:0(2Wdefault_policy/default_policy/sequential_2/twin_q_out/kernel/Adam_1/Initializer/zeros:0
Й
Adefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam:0Fdefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam/AssignUdefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam/Read/ReadVariableOp:0(2Sdefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam/Initializer/zeros:0
С
Cdefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam_1:0Hdefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam_1/AssignWdefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam_1/Read/ReadVariableOp:0(2Udefault_policy/default_policy/sequential_2/twin_q_out/bias/Adam_1/Initializer/zeros:0
Е
default_policy/beta1_power_3:0#default_policy/beta1_power_3/Assign2default_policy/beta1_power_3/Read/ReadVariableOp:0(28default_policy/beta1_power_3/Initializer/initial_value:0
Е
default_policy/beta2_power_3:0#default_policy/beta2_power_3/Assign2default_policy/beta2_power_3/Read/ReadVariableOp:0(28default_policy/beta2_power_3/Initializer/initial_value:0
э
.default_policy/default_policy/log_alpha/Adam:03default_policy/default_policy/log_alpha/Adam/AssignBdefault_policy/default_policy/log_alpha/Adam/Read/ReadVariableOp:0(2@default_policy/default_policy/log_alpha/Adam/Initializer/zeros:0
ѕ
0default_policy/default_policy/log_alpha/Adam_1:05default_policy/default_policy/log_alpha/Adam_1/AssignDdefault_policy/default_policy/log_alpha/Adam_1/Read/ReadVariableOp:0(2Bdefault_policy/default_policy/log_alpha/Adam_1/Initializer/zeros:0*ў
serving_defaultъ
1
is_training"
default_policy/is_training:0
 
D
observations4
default_policy/observation:0џџџџџџџџџ
8
seq_lens,
default_policy/seq_lens:0џџџџџџџџџ]
action_dist_inputsG
0default_policy/sequential_6/action_out/BiasAdd:0џџџџџџџџџ?
action_logp0
default_policy/cond_1/Merge:0џџџџџџџџџ8
action_prob)
default_policy/Exp_3:0џџџџџџџџџ?
	actions_02
default_policy/cond/Merge:0џџџџџџџџџtensorflow/serving/predict