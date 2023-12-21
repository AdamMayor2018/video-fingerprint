# Video deduplication with Distill-and-Select

*author: Chen Zhang*


<br />




## Description

This operator is made for video deduplication task base on [DnS: Distill-and-Select for Efficient and Accurate Video Indexing and Retrieval](https://arxiv.org/abs/2106.13266).  
Training with knowledge distillation method in large, unlabelled datasets, DnS learns: a) Student Networks at different retrieval performance and computational efficiency trade-offs and b) a Selection Network that at test time rapidly directs samples to the appropriate student to maintain both high retrieval performance and high computational efficiency.

![](DnS.png)

<br />


## Code Example

Load a video from path './demo_video.flv' using ffmpeg operator to decode it.

Then use distill_and_select operator to get the output using the specified model.

For fine-grained student model, get a 3d output with the temporal-dim information. For coarse-grained student model, get a 1d output representing the whole video. For selector model, get a scalar output.

 *For feature_extractor model*:

```python
from towhee import pipe, ops, DataCollection

p = (
    pipe.input('video_path') \
        .map('video_path', 'video_gen', ops.video_decode.ffmpeg(start_time=0.0, end_time=1000.0, sample_type='time_step_sample', args={'time_step': 1})) \
        .map('video_gen', 'video_list', lambda x: [y for y in x]) \
        .map('video_list', 'vec',  ops.video_copy_detection.distill_and_select(model_name='feature_extractor', device='cuda:0')) \
        .output('vec')
)

DataCollection(p('./demo_video.flv')).show()
```
![](output_imgs/feature_extractor.png)


 *For fg_att_student model*:

```python
from towhee import pipe, ops, DataCollection

p = (
    pipe.input('video_path') \
        .map('video_path', 'video_gen', ops.video_decode.ffmpeg(start_time=0.0, end_time=1000.0, sample_type='time_step_sample', args={'time_step': 1})) \
        .map('video_gen', 'video_list', lambda x: [y for y in x]) \
        .map('video_list', 'vec',  ops.video_copy_detection.distill_and_select(model_name='fg_att_student', device='cuda:0')) \
        .output('vec')
)

DataCollection(p('./demo_video.flv')).show()
```
![](output_imgs/fg_att_student.png)


 *For fg_bin_student model*:

```python
from towhee import pipe, ops, DataCollection

p = (
    pipe.input('video_path') \
        .map('video_path', 'video_gen', ops.video_decode.ffmpeg(start_time=0.0, end_time=1000.0, sample_type='time_step_sample', args={'time_step': 1})) \
        .map('video_gen', 'video_list', lambda x: [y for y in x]) \
        .map('video_list', 'vec',  ops.video_copy_detection.distill_and_select(model_name='fg_bin_student', device='cuda:0')) \
        .output('vec')
)

DataCollection(p('./demo_video.flv')).show()
```
![](output_imgs/fg_bin_student.png)


 *For cg_student model*:

```python
from towhee import pipe, ops, DataCollection

p = (
    pipe.input('video_path') \
        .map('video_path', 'video_gen', ops.video_decode.ffmpeg(start_time=0.0, end_time=1000.0, sample_type='time_step_sample', args={'time_step': 1})) \
        .map('video_gen', 'video_list', lambda x: [y for y in x]) \
        .map('video_list', 'vec',  ops.video_copy_detection.distill_and_select(model_name='cg_student', device='cuda:0')) \
        .output('vec')
)

DataCollection(p('./demo_video.flv')).show()
```
![](output_imgs/cg_student.png)


 *For selector_att model*:

```python
from towhee import pipe, ops, DataCollection

p = (
    pipe.input('video_path') \
        .map('video_path', 'video_gen', ops.video_decode.ffmpeg(start_time=0.0, end_time=1000.0, sample_type='time_step_sample', args={'time_step': 1})) \
        .map('video_gen', 'video_list', lambda x: [y for y in x]) \
        .map('video_list', 'vec',  ops.video_copy_detection.distill_and_select(model_name='selector_att', device='cuda:0')) \
        .output('vec')
)

DataCollection(p('./demo_video.flv')).show()
```
![](output_imgs/selector_att.png)


 *For selector_bin model*:

```python
from towhee import pipe, ops, DataCollection

p = (
    pipe.input('video_path') \
        .map('video_path', 'video_gen', ops.video_decode.ffmpeg(start_time=0.0, end_time=1000.0, sample_type='time_step_sample', args={'time_step': 1})) \
        .map('video_gen', 'video_list', lambda x: [y for y in x]) \
        .map('video_list', 'vec',  ops.video_copy_detection.distill_and_select(model_name='selector_bin', device='cuda:0')) \
        .output('vec')
)

DataCollection(p('./demo_video.flv')).show()
```
![](output_imgs/selector_bin.png)



<br />



## Factory Constructor

Create the operator via the following factory method

***distill_and_select(model_name, \*\*kwargs)***

**Parameters:**

​   ***model_name:*** *str*

​  Can be one of them:  
`feature_extractor`: Feature Extractor only,  
`fg_att_student`: Fine Grained Student with attention,  
`fg_bin_student`: Fine Grained Student with binarization,  
`cg_student`: Coarse Grained Student,  
`selector_att`: Selector Network with attention,  
`selector_bin`: Selector Network with binarization.  


​   ***model_weight_path:*** *str*

​   Default is None, download use the original pretrained weights. 

​   ***feature_extractor:*** *Union[str, nn.Module]*

​   `None`, 'default' or a pytorch nn.Module instance.  
`None` means this operator don't support feature extracting from the video data and this operator process embedding feature as input.  
'default' means using the original pretrained feature extracting weights and this operator can process video data as input.  
Or you can pass in a nn.Module instance as a specified feature extractor.  
Default is `default`.

​   ***device:*** *str*  
​  Model device, cpu or cuda.

<br />



## Interface

Get the output from your specified model.

**Parameters:**

​	***data:*** *List[towhee.types.VideoFrame]*  or *Any*

​  The input type is List[VideoFrame] when using default feature_extractor, else the type for your customer feature_extractor. 
	



**Returns:** *numpy.ndarray*

​  Output by specified model.




