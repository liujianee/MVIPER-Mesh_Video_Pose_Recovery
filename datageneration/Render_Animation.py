import sys
import os
import random
from math import radians
import bpy
import numpy as np
from os import getenv
from os import remove
from os.path import join, dirname, realpath, exists
from mathutils import Matrix, Vector, Quaternion, Euler
from glob import glob
from random import choice
from pickle import load
from bpy_extras.object_utils import world_to_camera_view as world2cam

sys.path.insert(0, ".")
#sys.path.append('/home/jianl/virenv_python35/lib/python3.5/site-packages/')
sys.path.append('/home/jianl/miniconda2/envs/STGCN/lib/python3.5/site-packages/')

import scipy.io as sio
import re
import cv2

PREAMBLE_FRAME_NUM = 10

CAMERA_DEPTH = 125#100#35
CAMERA_HEIGHT = 10	# in cam_mode=3, change HEIGHT to 0

SCALE_FAC = 10

cam_mode = 1	# TODO
frame_only_mode = False
#frame_only_mode = True

def mkdir_safe(directory):
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass

def setState0():
    for ob in bpy.data.objects.values():
        ob.select=False
    bpy.context.scene.objects.active = None

sorted_parts = ['hips','leftUpLeg','rightUpLeg','spine','leftLeg','rightLeg',
                'spine1','leftFoot','rightFoot','spine2','leftToeBase','rightToeBase',
                'neck','leftShoulder','rightShoulder','head','leftArm','rightArm',
                'leftForeArm','rightForeArm','leftHand','rightHand','leftHandIndex1' ,'rightHandIndex1']
# order
# part_match = {'root':'root', 'bone_00':'Pelvis', 'bone_01':'L_Hip', 'bone_02':'R_Hip',
#               'bone_03':'Spine1', 'bone_04':'L_Knee', 'bone_05':'R_Knee', 'bone_06':'Spine2',
#               'bone_07':'L_Ankle', 'bone_08':'R_Ankle', 'bone_09':'Spine3', 'bone_10':'L_Foot',
#               'bone_11':'R_Foot', 'bone_12':'Neck', 'bone_13':'L_Collar', 'bone_14':'R_Collar',
#               'bone_15':'Head', 'bone_16':'L_Shoulder', 'bone_17':'R_Shoulder', 'bone_18':'L_Elbow',
#               'bone_19':'R_Elbow', 'bone_20':'L_Wrist', 'bone_21':'R_Wrist', 'bone_22':'L_Hand', 'bone_23':'R_Hand'}

# Use 'Head_end', rather than 'Head'
part_match = {'root':'root', 'bone_00':'Pelvis', 'bone_01':'L_Hip', 'bone_02':'R_Hip',
              'bone_03':'Spine1', 'bone_04':'L_Knee', 'bone_05':'R_Knee', 'bone_06':'Spine2',
              'bone_07':'L_Ankle', 'bone_08':'R_Ankle', 'bone_09':'Spine3', 'bone_10':'L_Foot',
              'bone_11':'R_Foot', 'bone_12':'Neck', 'bone_13':'L_Collar', 'bone_14':'R_Collar',
              'bone_15':'Head_end', 'bone_16':'L_Shoulder', 'bone_17':'R_Shoulder', 'bone_18':'L_Elbow',
              'bone_19':'R_Elbow', 'bone_20':'L_Wrist', 'bone_21':'R_Wrist', 'bone_22':'L_Hand', 'bone_23':'R_Hand'}

part2num = {part:(ipart+1) for ipart,part in enumerate(sorted_parts)}

# create one material per part as defined in a pickle with the segmentation
# this is useful to render the segmentation in a material pass
def create_segmentation(ob, params):
    materials = {}
    vgroups = {}
    with open('pkl/segm_per_v_overlap.pkl', 'rb') as f:
        vsegm = load(f)
    bpy.ops.object.material_slot_remove()
    parts = sorted(vsegm.keys())
    for part in parts:
        vs = vsegm[part]
        vgroups[part] = ob.vertex_groups.new(part)
        vgroups[part].add(vs, 1.0, 'ADD')
        bpy.ops.object.vertex_group_set_active(group=part)
        materials[part] = bpy.data.materials['Material'].copy()
        materials[part].pass_index = part2num[part]
        bpy.ops.object.material_slot_add()
        ob.material_slots[-1].material = materials[part]
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.vertex_group_select()
        bpy.ops.object.material_slot_assign()
        bpy.ops.object.mode_set(mode='OBJECT')
    return(materials)


def create_hdr_background(img=None):
    bpy.context.scene.cycles.film_transparent = False
    bpy.context.scene.world.use_nodes = True
    tree = bpy.context.scene.world.node_tree
    bg = tree.nodes['Background']	# default node
    wo = tree.nodes['World Output']	# default node
    tex_coord = tree.nodes.new('ShaderNodeTexCoord')
    map_ing = tree.nodes.new('ShaderNodeMapping')
    env_tex = tree.nodes.new('ShaderNodeTexEnvironment')
    tex_coord.location = -1200, 300
    map_ing.location = -800, 300
    env_tex.location = -300, 300
    if img is not None:
        env_tex.image = img
    tree.links.new(tex_coord.outputs[0], map_ing.inputs[0])
    tree.links.new(map_ing.outputs[0], env_tex.inputs[0])
    tree.links.new(env_tex.outputs[0], bg.inputs[0])


def change_hdr_background(img=None, resize_fac=1., rotate_z=0., view_m=0):
    tree = bpy.context.scene.world.node_tree
    env_tex = tree.nodes['Environment Texture']
    map_ing = tree.nodes['Mapping']
    if img is not None:
        env_tex.image = img
        if view_m == 0 or view_m == 3:
            map_ing.scale = Vector([resize_fac, 1, resize_fac])
        elif view_m == 1 or view_m == 2:
            map_ing.scale = Vector([1, resize_fac, resize_fac])
        map_ing.rotation[2] = rotate_z


def random_hdr_background(hdr_path, hdr_files, view_mode):
    hdr_img_name = join(hdr_path, choice(hdr_files))
    hdr_img = bpy.data.images.load(hdr_img_name)
    hdr_resize_fac  = np.random.normal(10, 1)
    hdr_rotate_z = np.random.uniform(0, 6.3)
    if cam_mode == 1 or cam_mode == 2 or cam_mode == 3:
        change_hdr_background(img=hdr_img, resize_fac=hdr_resize_fac, rotate_z=hdr_rotate_z, view_m=view_mode)

# create the different passes that we render
def create_composite_nodes(tree, params, img=None, idx=0):
    #https://blender.stackexchange.com/questions/40648/cycles-not-rendering-with-background-image
    bpy.context.scene.cycles.film_transparent = True
    res_paths = {k:join(params['tmp_path'], '%05d_%s'%(idx, k)) for k in params['output_types'] if params['output_types'][k]}
    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)
    # create node for foreground image
    layers = tree.nodes.new('CompositorNodeRLayers')
    layers.location = -300, 400
    # create node for background image
    bg_im = tree.nodes.new('CompositorNodeImage')
    bg_im.location = -300, 30
    if img is not None:
        bg_im.image = img
    if(params['output_types']['vblur']):
    # create node for computing vector blur (approximate motion blur)
        vblur = tree.nodes.new('CompositorNodeVecBlur')
        vblur.factor = params['vblur_factor']
        vblur.location = 240, 400
        # create node for saving output of vector blurred image
        vblur_out = tree.nodes.new('CompositorNodeOutputFile')
        vblur_out.format.file_format = 'PNG'
        vblur_out.base_path = res_paths['vblur']
        vblur_out.location = 460, 460
    # create node for mixing foreground and background images
    mix = tree.nodes.new('CompositorNodeMixRGB')
    mix.location = 40, 30
    mix.use_alpha = True
    # create node for the final output
    composite_out = tree.nodes.new('CompositorNodeComposite')
    composite_out.location = 240, 30
    # create node for saving depth
    if(params['output_types']['depth']):
        depth_out = tree.nodes.new('CompositorNodeOutputFile')
        depth_out.location = 40, 700
        depth_out.format.file_format = 'OPEN_EXR'
        depth_out.base_path = res_paths['depth']
    # create node for saving normals
    if(params['output_types']['normal']):
        normal_out = tree.nodes.new('CompositorNodeOutputFile')
        normal_out.location = 40, 600
        normal_out.format.file_format = 'OPEN_EXR'
        normal_out.base_path = res_paths['normal']
    # create node for saving foreground image
    if(params['output_types']['fg']):
        fg_out = tree.nodes.new('CompositorNodeOutputFile')
        fg_out.location = 170, 600
        fg_out.format.file_format = 'PNG'
        fg_out.base_path = res_paths['fg']
    # create node for saving ground truth flow
    if(params['output_types']['gtflow']):
        gtflow_out = tree.nodes.new('CompositorNodeOutputFile')
        gtflow_out.location = 40, 500
        gtflow_out.format.file_format = 'OPEN_EXR'
        gtflow_out.base_path = res_paths['gtflow']
    # create node for saving segmentation
    if(params['output_types']['segm']):
        segm_out = tree.nodes.new('CompositorNodeOutputFile')
        segm_out.location = 40, 400
        segm_out.format.file_format = 'OPEN_EXR'
        segm_out.base_path = res_paths['segm']
    # merge fg and bg images
    tree.links.new(bg_im.outputs[0], mix.inputs[1])
    tree.links.new(layers.outputs['Image'], mix.inputs[2])
    if(params['output_types']['vblur']):
        tree.links.new(mix.outputs[0], vblur.inputs[0])                # apply vector blur on the bg+fg image,
        tree.links.new(layers.outputs['Z'], vblur.inputs[1])           #   using depth,
        tree.links.new(layers.outputs['Speed'], vblur.inputs[2])       #   and flow.
        tree.links.new(vblur.outputs[0], vblur_out.inputs[0])          # save vblurred output
    tree.links.new(mix.outputs[0], composite_out.inputs[0])            # bg+fg image
    if(params['output_types']['fg']):
        tree.links.new(layers.outputs['Image'], fg_out.inputs[0])      # save fg
    if(params['output_types']['depth']):
        tree.links.new(layers.outputs['Z'], depth_out.inputs[0])       # save depth
    if(params['output_types']['normal']):
        tree.links.new(layers.outputs['Normal'], normal_out.inputs[0]) # save normal
    if(params['output_types']['gtflow']):
        tree.links.new(layers.outputs['Speed'], gtflow_out.inputs[0])  # save ground truth flow
    if(params['output_types']['segm']):
        tree.links.new(layers.outputs['IndexMA'], segm_out.inputs[0])  # save segmentation
    return(res_paths)

# creation of the spherical harmonics material, using an OSL script
def create_sh_material(tree, sh_path, img=None):
    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)
    uv = tree.nodes.new('ShaderNodeTexCoord')
    uv.location = -800, 400
    uv_xform = tree.nodes.new('ShaderNodeVectorMath')
    uv_xform.location = -600, 400
    uv_xform.inputs[1].default_value = (0, 0, 1)
    uv_xform.operation = 'AVERAGE'
    uv_im = tree.nodes.new('ShaderNodeTexImage')
    uv_im.location = -400, 400
    if img is not None:
        uv_im.image = img
    rgb = tree.nodes.new('ShaderNodeRGB')
    rgb.location = -400, 200
    script = tree.nodes.new('ShaderNodeScript')
    script.location = -230, 400
    script.mode = 'EXTERNAL'
    script.filepath = sh_path #'spher_harm/sh.osl' #using the same file from multiple jobs causes white texture
    script.update()
    # the emission node makes it independent of the scene lighting
    emission = tree.nodes.new('ShaderNodeEmission')
    emission.location = -60, 400
    mat_out = tree.nodes.new('ShaderNodeOutputMaterial')
    mat_out.location = 110, 400
    tree.links.new(uv.outputs[2], uv_im.inputs[0])
    tree.links.new(uv_im.outputs[0], script.inputs[0])
    tree.links.new(script.outputs[0], emission.inputs[0])
    tree.links.new(emission.outputs[0], mat_out.inputs[0])

def create_model_material(tree, img=None):
    diffuse = tree.nodes['Diffuse BSDF']
    mo = tree.nodes['Material Output']
    img_tex = tree.nodes.new('ShaderNodeTexImage')
    img_tex.location = -200, 300
    if img is not None:
        img_tex.image = img
    tree.links.new(img_tex.outputs[0], diffuse.inputs[0])

# only recall this if the cloth is not unwrapped from previous step (MD7)
# this ususally happens if no texture is assigned to fabric in MD7
def cloth_unwrap(cloth_ob):
    bpy.context.scene.objects.active = cloth_ob
    #bpy.ops.object.editmode_toggle()
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)
    #bpy.ops.object.editmode_toggle()
    bpy.ops.object.mode_set(mode='OBJECT')

# creation of the fabric material
def create_fabric_material(tree, img=None):
    # clear default nodes
    #for n in tree.nodes:
    #    tree.nodes.remove(n)
    mat_out = tree.nodes['Material Output']
    diffuse = tree.nodes['Diffuse BSDF']
    img_tex = tree.nodes.new('ShaderNodeTexImage')
    tex_coord = tree.nodes.new('ShaderNodeTexCoord')
    map_ing = tree.nodes.new('ShaderNodeMapping')
    img_tex.location = -200, 300
    tex_coord.location = -800, 300
    map_ing.location = -600, 300
    if img is not None:
        img_tex.image = img
    tree.links.new(img_tex.outputs[0], diffuse.inputs[0])
    tree.links.new(tex_coord.outputs[2], map_ing.inputs[0])
    tree.links.new(map_ing.outputs[0], img_tex.inputs[0])

def change_fabric_material(tree, img=None, resize_fac=1.):
    img_tex = tree.nodes['Image Texture']
    map_ing = tree.nodes['Mapping']
    if img is not None:
        img_tex.image = img
        map_ing.scale = Vector([resize_fac, resize_fac, 1])

# computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
    return(cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)

def init_scene(body_object, body_motion, cloth_object, cloth_motion, scene, params, gender='female'):
    # load body object
    bpy.ops.import_scene.obj(filepath=body_object, split_mode='OFF')
    obname = body_object.split('/')[-1].split('.')[0]
    ob = bpy.data.objects[obname]
    scn = bpy.context.scene
    scn.objects.active = ob
    # Add modifier for motion
    bpy.ops.object.modifier_add(type='MESH_CACHE')
    ob.modifiers['Mesh Cache'].filepath = body_motion
    ob.modifiers['Mesh Cache'].forward_axis = 'NEG_Z'
    ob.modifiers['Mesh Cache'].up_axis = 'POS_Y'
    # load cloth object
    bpy.ops.import_scene.obj(filepath=cloth_object, split_mode='OFF')
    #cloth_obname = body_object.split('/')[-1].split('.')[0] + '.001'
    cloth_obname = cloth_object.split('/')[-1].split('.')[0]
    cloth_ob = bpy.data.objects[cloth_obname]
    scn.objects.active = cloth_ob
    # Add modifier for motion
    bpy.ops.object.modifier_add(type='MESH_CACHE')
    cloth_ob.modifiers['Mesh Cache'].filepath = cloth_motion
    cloth_ob.modifiers['Mesh Cache'].flip_axis = {'Z'}
    #ob.data.use_auto_smooth = False  # autosmooth creates artifacts
    # assign the existing spherical harmonics material
    ob.active_material = bpy.data.materials['Material']
    # delete the default cube (which held the material)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete(use_global=False)
    bpy.data.objects['Lamp'].select = True
    bpy.ops.object.delete(use_global=False)
    # set camera properties and initial position
    bpy.ops.object.select_all(action='DESELECT')
    cam_ob = bpy.data.objects['Camera']
    scn.objects.active = cam_ob
    cam_ob.matrix_world = Matrix(((1., 0., 0, 0),(0., 0, -1., -CAMERA_DEPTH),(0., 1., 0., CAMERA_HEIGHT),(0.0, 0.0, 0.0, 1.0)))
    cam_ob.data.lens =  180#120#35	# unit: mm
    cam_ob.data.sensor_width = 32
    # parent camera to a empty at the origin
    bpy.ops.object.empty_add(type='PLAIN_AXES', radius=1, view_align=False, location=(0, 0, 0))
    #bpy.data.objects['Camera'].select = True
    #bpy.ops.object.parent_set(type='OBJECT', keep_transform=False)
    #scn.cycles.film_transparent = True
    #scn.cycles.film_transparent = False	# ATTENTION: if this was set True, HDR background will not be showing
    scn.render.layers["RenderLayer"].use_pass_vector = True
    scn.render.layers["RenderLayer"].use_pass_normal = True
    scene.render.layers['RenderLayer'].use_pass_emit  = True
    scene.render.layers['RenderLayer'].use_pass_emit  = True
    scene.render.layers['RenderLayer'].use_pass_material_index  = True
    # set render size
    scn.render.resolution_x = params['resy']
    scn.render.resolution_y = params['resx']
    scn.render.resolution_percentage = 100
    scn.render.image_settings.file_format = 'PNG'
    # clear existing animation data
    #ob.data.shape_keys.animation_data_clear()
    #arm_ob = bpy.data.objects['Armature']
    #arm_ob.animation_data_clear()
    #return(ob, obname, arm_ob, cam_ob)
    return(ob, obname, cam_ob, cloth_ob, cloth_obname)

def import_armature(armature_on, body_armature, scale_factor=1.0):
    if armature_on is True:
        bpy.ops.import_scene.fbx(filepath=body_armature, global_scale=scale_factor)
        arm_ob = bpy.data.objects['Armature']
    else:
        arm_ob = None
    return(arm_ob)

def scale_objects(obj=None, scale_factor=1.0):
    obj.scale = Vector([scale_factor, scale_factor, scale_factor])

def camera_mode_a(cam_ob):
    bpy.context.scene.objects.active = cam_ob
    cam_ob.matrix_world = Matrix(((1., 0., 0, 0),(0., 0, -1., -CAMERA_DEPTH),(0., 1., 0., CAMERA_HEIGHT),(0.0, 0.0, 0.0, 1.0)))
    #cam_ob.data.angle = radians(40)	# this is field of view
    cam_ob.data.lens =  180#120#35
    cam_ob.data.clip_start = 0.1
    cam_ob.data.clip_end = 240
    cam_ob.data.sensor_width = 32

def camera_mode_b(cam_ob, arm_ob):
    bpy.context.scene.objects.active = cam_ob
    bpy.ops.object.constraint_add(type='TRACK_TO')	# think of "LOCKED_TRACK" if one axis needs to be fixed
    bpy.context.object.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
    bpy.context.object.constraints["Track To"].up_axis = 'UP_Y'
    bpy.context.object.constraints["Track To"].target = arm_ob #bpy.data.objects["Armature"]
    arm_subtarget = [n for n in arm_ob.data.bones.keys() if 'Pelvis' in n][0]
    bpy.context.object.constraints["Track To"].subtarget = arm_subtarget

def camera_mode_c(cam_ob, arm_ob):
    arm_subtarget = [n for n in arm_ob.data.bones.keys() if 'Pelvis' in n][0]
    arm_ob.data.bones.active = arm_ob.data.bones[arm_subtarget]
    cam_ob.select = True
    bpy.context.scene.objects.active = arm_ob
    bpy.ops.object.parent_set(type='OBJECT', keep_transform=False)
    bpy.ops.object.parent_set(type='BONE')

#def camera_mode_d(cam_ob, arm_ob):
#    #bpy.ops.object.parent_set(type='OBJECT', keep_transform=False)
#    cam_ob.parent = arm_ob


def camera_viewpoint(cam_ob, cam_parent, viewpoint):
    cam_parent.rotation_euler[0] = radians(viewpoint[0])
    cam_parent.rotation_euler[1] = radians(viewpoint[1])
    cam_parent.rotation_euler[2] = radians(viewpoint[2])
    # update() is necessary to make "cam_parent.matrix_world" updated
    bpy.data.scenes['Scene'].update()
    #print(cam_parent.rotation_euler)
    #print(cam_ob.matrix_world)
    cam_ob.matrix_world = cam_parent.matrix_world * cam_ob.matrix_world
    #print(cam_parent.matrix_world)
    #print(cam_ob.matrix_world)

# modify global rotation of smpl pose
def smpl_pose_offset(smpl_pose, viewpoint):
    root_rot_axis = smpl_pose	# axis-angle rotation for 'root'
    root_rot_euler = cv2.Rodrigues(root_rot_axis)[0] # euler rotation
    # counter-rotation matrix relative to the camera
    rot_matrix = Euler((radians(-viewpoint[0]), 0., radians(-viewpoint[2])), 'ZYX').to_matrix()
    new_root_rot_euler = np.matmul(rot_matrix, root_rot_euler) # euler rotation
    new_root_rot_axis = cv2.Rodrigues(new_root_rot_euler)[0] # convert back to axis-angle representation
    return new_root_rot_axis[:,0]	# aligned axis-angle rotation for 'root'

def lamp_setting(lamp, size, location, rotation, strength):
    #lamp_0 = bpy.data.objects['Lamp']
    #scene.objects.active = lamp_0
    lamp.data.type = "AREA"
    lamp.data.size = size
    lamp.location = Vector(location)
    lamp.rotation_euler[0] = radians(rotation[0])
    lamp.rotation_euler[1] = radians(rotation[1])
    lamp.rotation_euler[2] = radians(rotation[2])
    bpy.data.lamps[lamp.name].use_nodes = True
    lamp.data.node_tree.nodes['Emission'].inputs[1].default_value = strength


# transformation between pose and blendshapes
def rodrigues2bshapes(pose):
    rod_rots = np.asarray(pose).reshape(24, 3)
    mat_rots = [Rodrigues(rod_rot) for rod_rot in rod_rots]
    bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel()
                              for mat_rot in mat_rots[1:]])
    return(mat_rots, bshapes)

# apply trans pose and shape to character
def apply_trans_pose_shape(trans, pose, shape, ob, arm_ob, obname, scene, cam_ob, frame=None):
    # transform pose into rotation matrices (for pose) and pose blendshapes
    mrots, bsh = rodrigues2bshapes(pose)
    # set the location of the first bone to the translation parameter
    arm_ob.pose.bones[obname+'_Pelvis'].location = trans
    if frame is not None:
        arm_ob.pose.bones[obname+'_root'].keyframe_insert('location', frame=frame)
    # set the pose of each bone to the quaternion specified by pose
    for ibone, mrot in enumerate(mrots):
        bone = arm_ob.pose.bones[obname+'_'+part_match['bone_%02d' % ibone]]
        bone.rotation_quaternion = Matrix(mrot).to_quaternion()
        if frame is not None:
            bone.keyframe_insert('rotation_quaternion', frame=frame)
            bone.keyframe_insert('location', frame=frame)
    # apply pose blendshapes
    for ibshape, bshape in enumerate(bsh):
        ob.data.shape_keys.key_blocks['Pose%03d' % ibshape].value = bshape
        if frame is not None:
            ob.data.shape_keys.key_blocks['Pose%03d' % ibshape].keyframe_insert('value', index=-1, frame=frame)
    # apply shape blendshapes
    for ibshape, shape_elem in enumerate(shape):
        ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].value = shape_elem
        if frame is not None:
            ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].keyframe_insert('value', index=-1, frame=frame)


def get_bone_locs(arm_name, arm_ob, scene, cam_ob):
    n_bones = 24
    render_scale = scene.render.resolution_percentage / 100
    render_size = (int(scene.render.resolution_x * render_scale),
                   int(scene.render.resolution_y * render_scale))
    bone_locations_2d = np.empty((n_bones, 2))
    bone_locations_3d = np.empty((n_bones, 3), dtype='float32')
    # obtain the coordinates of each bone head in image space
    for ibone in range(n_bones):
        bone = arm_ob.pose.bones[arm_name+'_'+part_match['bone_%02d' % ibone]]
        co_2d = world2cam(scene, cam_ob, arm_ob.matrix_world * bone.head)
        co_3d = arm_ob.matrix_world * bone.head
        bone_locations_3d[ibone] = (co_3d.x,
                                 co_3d.y,
                                 co_3d.z)
        bone_locations_2d[ibone] = (round(co_2d.x * render_size[0]),
                                 round(co_2d.y * render_size[1]))
    return(bone_locations_2d, bone_locations_3d)


# reset the joint positions of the character according to its new shape
def reset_joint_positions(orig_trans, shape, ob, arm_ob, obname, scene, cam_ob, reg_ivs, joint_reg):
    # since the regression is sparse, only the relevant vertex
    #     elements (joint_reg) and their indices (reg_ivs) are loaded
    reg_vs = np.empty((len(reg_ivs), 3))  # empty array to hold vertices to regress from
    # zero the pose and trans to obtain joint positions in zero pose
    apply_trans_pose_shape(orig_trans, np.zeros(72), shape, ob, arm_ob, obname, scene, cam_ob)
    # obtain a mesh after applying modifiers
    bpy.ops.wm.memory_statistics()
    # me holds the vertices after applying the shape blendshapes
    me = ob.to_mesh(scene, True, 'PREVIEW')
    # fill the regressor vertices matrix
    for iiv, iv in enumerate(reg_ivs):
        reg_vs[iiv] = me.vertices[iv].co
    bpy.data.meshes.remove(me)
    # regress joint positions in rest pose
    joint_xyz = joint_reg.dot(reg_vs)
    # adapt joint positions in rest pose
    arm_ob.hide = False
    bpy.ops.object.mode_set(mode='EDIT')
    arm_ob.hide = True
    for ibone in range(24):
        bb = arm_ob.data.edit_bones[obname+'_'+part_match['bone_%02d' % ibone]]
        bboffset = bb.tail - bb.head
        bb.head = joint_xyz[ibone]
        bb.tail = bb.head + bboffset
    bpy.ops.object.mode_set(mode='OBJECT')
    return(shape)

# load poses and shapes
def load_body_data(smpl_data, ob, obname, gender='female', idx=0):
    # load MoSHed data from CMU Mocap (only the given idx is loaded)
    # create a dictionary with key the sequence name and values the pose and trans
    cmu_keys = []
    for seq in smpl_data.files:
        if seq.startswith('pose_'):
            cmu_keys.append(seq.replace('pose_', ''))
    name = sorted(cmu_keys)[idx % len(cmu_keys)]
    cmu_parms = {}
    for seq in smpl_data.files:
        if seq == ('pose_' + name):
            cmu_parms[seq.replace('pose_', '')] = {'poses':smpl_data[seq],
                                                   'trans':smpl_data[seq.replace('pose_','trans_')]}
    # compute the number of shape blendshapes in the model
    n_sh_bshapes = len([k for k in ob.data.shape_keys.key_blocks.keys()
                        if k.startswith('Shape')])
    # load all SMPL shapes
    fshapes = smpl_data['%sshapes' % gender][:, :n_sh_bshapes]
    return(cmu_parms, fshapes, name)


def collect_fabrics(fabric_path):
    fab_files = []
    fab_num = 500
    for one_path in fabric_path:
      if 'cloth' in one_path:
        fab_files_1 = glob(join(one_path, '*.png'))
        fab_files_1.extend(glob(join(one_path, '*.jpg')))
        fab_num = len(fab_files_1)
      if 'dtd' in one_path:
        fab_files_2 = []
        for root, dirs, files in os.walk(join(one_path, 'images')):
          if files is not None:
            fab_files_2.extend(glob(join(root, '*.jpg')))
        shuffle_id = np.random.permutation(len(fab_files_2))
        fab_files_2 = list(np.asarray(fab_files_2)[shuffle_id[0:fab_num]])
      if 'Fabrics' in one_path:
        fab_files_3 = []
        for root, dirs, files in os.walk(one_path):
          if files is not None:
            fab_files_3.extend(glob(join(root, '*.png')))
        shuffle_id = np.random.permutation(len(fab_files_3))
        fab_files_3 = list(np.asarray(fab_files_3)[shuffle_id[0:fab_num]])
    return fab_files_1 + fab_files_2 + fab_files_3


import time
start_time = 0.
def log_message(message):
    elapsed_time = time.time() - start_time
    print("[%.2f s] %s" % (elapsed_time, message))

def main(runpass=None, idx=None, stride=None):
    # time logging
    global start_time
    start_time = time.time()
    import argparse

    #idx = 3
    #ishape = 0
    #stride = 50

    # # parse commandline arguments
    # log_message(sys.argv)
    # parser = argparse.ArgumentParser(description='Generate synth dataset images.')
    # parser.add_argument('--idx', type=int,
    #                     help='idx of the requested sequence')
    # parser.add_argument('--ishape', type=int,
    #                     help='requested cut, according to the stride')
    # parser.add_argument('--stride', type=int,
    #                     help='stride amount, default 50')
    #
    # args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    #
    # idx = args.idx
    # ishape = args.ishape
    # stride = args.stride

    if idx == None:
        log_message("WARNING: idx not specified, using default value 0")
        idx = 0

    #if ishape == None:
    #    exit(1)

    if stride == None:
        log_message("WARNING: stride not specified, using default value 50")
        stride = 50

    log_message("input idx: %d" % idx)
    log_message("input stride: %d" % stride)

    # import idx info (name, split)
    idx_info = load(open("pkl/idx_info.pickle", 'rb'))

    # get runpass
    (_, idx) = divmod(idx, len(idx_info))

    log_message("runpass: %d" % runpass)
    log_message("output idx: %d" % idx)
    idx_info = idx_info[idx]
    log_message("sequence: %s" % idx_info['name'])
    log_message("nb_frames: %f" % idx_info['nb_frames'])
    log_message("use_split: %s" % idx_info['use_split'])

    # import configuration
    log_message("Importing configuration")
    import config
    params = config.load_file('config', 'SYNTH_DATA')

    smpl_data_folder = params['smpl_data_folder']
    smpl_data_filename = params['smpl_data_filename']
    bg_path = params['bg_path']
    resy = params['resy']
    resx = params['resx']
    clothing_option = params['clothing_option'] # grey, nongrey or all
    tmp_path = params['tmp_path']
    output_path = params['output_path']
    output_types = params['output_types']
    stepsize = params['stepsize']
    clipsize = params['clipsize']
    #openexr_py2_path = params['openexr_py2_path']

    # added by JIANL
    #garment_path = params['garment_path']
    render_output = params['render_output']
    fabric_path = params['fabric_path']
    hdr_path = params['hdr_path']


    # compute number of cuts
    #nb_ishape = max(1, int(np.ceil((idx_info['nb_frames'] - (clipsize - stride))/stride)))
    nb_ishape = 1
    log_message("Max ishape: %d" % (nb_ishape - 1))

    #if ishape == None:
    #    exit(1)

    #assert(ishape < nb_ishape)

    # name is set given idx
    #name = idx_info['name']
    #output_path = join(output_path, 'run%d' % runpass, name.replace(" ", ""))
    output_path = join(output_path, 'run%d' % runpass)
    params['output_path'] = output_path
    #tmp_path = join(tmp_path, 'run%d_%s_c%04d' % (runpass, name.replace(" ", ""), (ishape + 1)))
    #params['tmp_path'] = tmp_path

    fab_files = collect_fabrics(fabric_path)

    actions = os.listdir(output_path)
    for one_action in actions:
        name = one_action
        action_dir = join(output_path, one_action)
        clip_info = glob(join(action_dir, '*_info.mat'))
        clip_fbxs = glob(join(action_dir, '*.fbx'))
        clip_num = len(clip_fbxs) ########################
        for ishape in range(clip_num):
            garment_objs = glob(join(action_dir, '*_G*.obj'))
            for one_garment in garment_objs:
                garment_id = one_garment.split('_')[-1].split('.')[0]

                # check if already computed
                #  + clean up existing tmp folders if any
                #if exists(tmp_path) and tmp_path != "" and tmp_path != "/":
                #    os.system('rm -rf %s' % tmp_path)

                #rgb_vid_filename = "%s_c%04d.mp4" % (join(output_path, name.replace(' ', '')), (ishape + 1))
                #if os.path.isfile(rgb_vid_filename):
                #    log_message("ALREADY COMPUTED - existing: %s" % rgb_vid_filename)
                #    return 0

                # create tmp directory
                #if not exists(tmp_path):
                #    mkdir_safe(tmp_path)

                # >> don't use random generator before this point <<

                # initialize RNG with seeds from sequence id
                import hashlib
                #s = "synth_data:%d:%d:%d" % (idx, runpass,ishape)
                s = "synth_data:%d:%d:%d:%s" % (idx, runpass, ishape, garment_id)
                seed_number = int(hashlib.sha1(s.encode('utf-8')).hexdigest(), 16) % (10 ** 8)
                log_message("GENERATED SEED %d from string '%s'" % (seed_number, s))
                #random.seed(seed_number)
                #np.random.seed(seed_number)

                #if(output_types['vblur']):
                #    vblur_factor = np.random.normal(0.5, 0.5)
                #    params['vblur_factor'] = vblur_factor

                log_message("Setup Blender")

                # create copy-spher.harm. directory if not exists
                sh_dir = join(tmp_path, 'spher_harm')
                if not exists(sh_dir):
                    mkdir_safe(sh_dir)

                sh_dst = join(sh_dir, 'sh_%02d_%05d.osl' % (runpass, idx))
                os.system('cp spher_harm/sh.osl %s' % sh_dst)

                genders = {0: 'female', 1: 'male'}
                #garment_types = os.listdir(garment_path)

                #garment_type = choice(garment_types)
                #garment_path = join(garment_path, garment_type, 'run%d' % runpass, name.replace(" ", ""))

                # pick random gender
                #gender = choice(genders) ############################################################################

                scene = bpy.data.scenes['Scene']
                scene.render.engine = 'CYCLES'
                bpy.context.scene.cycles.device = 'GPU'
                scene.render.tile_x = 256
                scene.render.tile_y = 256

                #matfile_info = join(output_path, name.replace(" ", "") + "_c%04d_info.mat" % (ishape+1))
                matfile_info = clip_info[ishape]
                log_message('Reading annotation from %s' % matfile_info)
                annot_info = sio.loadmat(matfile_info)

                N = annot_info['gender'][0].shape[0]	#TODO

                gender = genders[annot_info['gender'][0][0]]
                #body_object = join(output_path, name.replace(" ", "") + "_c%04d.obj" % (ishape+1))
                body_object = clip_fbxs[ishape].replace('.fbx', '.obj')
                #body_motion = join(output_path, name.replace(" ", "") + "_c%04d.mdd" % (ishape+1))
                body_motion = clip_fbxs[ishape].replace('.fbx', '.mdd')

                armature_on = True	# TODO
                #body_armature = join(output_path, name.replace(" ", "") + "_c%04d.fbx" % (ishape+1))
                body_armature = clip_fbxs[ishape]

                #cloth_object = join(garment_path, name.replace(" ", "") + "_c%04d.obj" % (ishape+1))
                cloth_object = one_garment
                #cloth_motion = join(garment_path, name.replace(" ", "") + "_c%04d.mdd" % (ishape+1))
                cloth_motion = one_garment.replace('.obj', '.mdd')

                bpy.data.materials['Material'].use_nodes = True
                # DISABLE these two options, as it slows down the rendering
                #scene.cycles.shading_system = True	# set True to use OSL, but only with CPU rendering
                scene.use_nodes = True

                log_message("Listing background images")
                # bg_names = join(bg_path, '%s_img.txt' % idx_info['use_split'])
                nh_txt_paths = glob(join(bg_path, '*.jpg'))

                # nh_txt_paths = []
                # with open(bg_names) as f:
                #     for line in f:
                #         nh_txt_paths.append(join(bg_path, line))

                # grab clothing names
                log_message("clothing: %s" % clothing_option)
                with open( join(smpl_data_folder, 'textures', '%s_%s.txt' % ( gender, idx_info['use_split'] ) ) ) as f:
                    txt_paths = f.read().splitlines()

                # if using only one source of clothing
                if clothing_option == 'nongrey':
                    txt_paths = [k for k in txt_paths if 'nongrey' in k]
                elif clothing_option == 'grey':
                    txt_paths = [k for k in txt_paths if 'nongrey' not in k]

                # random clothing texture
                cloth_img_name = choice(txt_paths)
                cloth_img_name = join(smpl_data_folder, cloth_img_name)
                cloth_img = bpy.data.images.load(cloth_img_name)

                # non-HDR background
                bg_img_name = choice(nh_txt_paths)
                bg_img = bpy.data.images.load(bg_img_name)

                # HDR background
                hdr_files = os.listdir(hdr_path)
                hdr_img_name = join(hdr_path, choice(hdr_files))
                hdr_img = bpy.data.images.load(hdr_img_name)

                # fabric texture
                #fab_files = os.listdir(fabric_path)

                #log_message("Loading parts segmentation")
                #beta_stds = np.load(join(smpl_data_folder, ('%s_beta_stds.npy' % gender)))

                log_message("Building materials tree")
                mat_tree = bpy.data.materials['Material'].node_tree

                # STOP using OSL to mapping model texture, as it is SLOW to render
                #create_sh_material(mat_tree, sh_dst, cloth_img)
                create_model_material(mat_tree, cloth_img)


                # this function is to set up non-HDR background
                if cam_mode == 0:
                    res_paths = create_composite_nodes(bpy.context.scene.node_tree, params, img=bg_img, idx=idx)

                # this function is to set up HDR background
                # For cam_mode=1 and cam_mode=2, HDR background must be used
                if cam_mode == 1 or cam_mode == 2 or cam_mode == 3:
                    create_hdr_background(img=hdr_img)


                #log_message("Loading smpl data")
                #smpl_data = np.load(join(smpl_data_folder, smpl_data_filename))

                log_message("Initializing scene")
                # set up lamps
                log_message("Setting up lighting")
                bpy.ops.object.lamp_add(type='AREA')
                lamp_0 = bpy.data.objects['Area']
                lamp_0_siz = 10
                lamp_0_loc = [25,0,25]
                lamp_0_rot = [45,0,90]
                lamp_0_str = np.random.normal(5000, 2500)
                lamp_setting(lamp_0, lamp_0_siz, lamp_0_loc, lamp_0_rot, lamp_0_str)

                bpy.ops.object.lamp_add(type='AREA')
                lamp_1 = bpy.data.objects['Area.001']
                lamp_1_siz = 10
                lamp_1_loc = [-25,0,25]
                lamp_1_rot = [-45,0,90]
                lamp_1_str = np.random.normal(5000, 2500)
                lamp_setting(lamp_1, lamp_1_siz, lamp_1_loc, lamp_1_rot, lamp_1_str)

                bpy.ops.object.lamp_add(type='AREA')
                lamp_2 = bpy.data.objects['Area.002']
                lamp_2_siz = 10
                lamp_2_loc = [0,-25,25]
                lamp_2_rot = [45,0,0]
                lamp_2_str = np.random.normal(5000, 2500)
                lamp_setting(lamp_2, lamp_2_siz, lamp_2_loc, lamp_2_rot, lamp_2_str)

                bpy.ops.object.lamp_add(type='AREA')
                lamp_3 = bpy.data.objects['Area.003']
                lamp_3_siz = 10
                lamp_3_loc = [0,25,25]
                lamp_3_rot = [-45,0,0]
                lamp_3_str = np.random.normal(5000, 2500)
                lamp_setting(lamp_3, lamp_3_siz, lamp_3_loc, lamp_3_rot, lamp_3_str)

                #camera_distance = np.random.normal(8.0, 1)
                #camera_distance = CAMERA_DISTANCE
                #params['camera_distance'] = camera_distance ###############################################################
                ob, obname, cam_ob, cloth_ob, cloth_obname = init_scene(body_object, body_motion, cloth_object, cloth_motion, scene, params, gender)
                bpy.context.scene.frame_start = PREAMBLE_FRAME_NUM	# exclude the preambles during rendering
                bpy.context.scene.frame_end = N
                cam_parent = bpy.data.objects['Empty']

                # scale up objects if needed
                scale_objects(ob, scale_factor=SCALE_FAC)
                scale_objects(cloth_ob, scale_factor=SCALE_FAC)

                # texture mapping for garments
                #scene.objects.active = cloth_ob
                #bpy.ops.object.editmode_toggle()
                #bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)
                #bpy.ops.object.editmode_toggle()
                # TODO: NO need to unwrap if fabirc(texture) has been added in MD7ac)
                #cloth_unwrap(cloth_ob)


                #fabric_patterns = [fab for fab in bpy.data.materials.keys() if 'FABRIC' in fab]
                fabric_patterns = [fab for fab in bpy.data.materials.keys() if re.search('Material[0-9]+', fab) or 'FABRIC' in fab]	# TODO
                assert(len(fabric_patterns) > 0)

                for one_fabric in fabric_patterns:
                    fab_mat = bpy.data.materials[one_fabric]
                    fab_mat.use_nodes = True
                    fab_mat_tree = fab_mat.node_tree
                    #fab_img_name = join(fabric_path, choice(fab_files))	# randomly select fabric texture
                    fab_img_name = choice(fab_files)	# randomly select fabric texture
                    fab_img = bpy.data.images.load(fab_img_name)
                    create_fabric_material(fab_mat_tree, fab_img)

                # import armature for camera setting and rendering annotation
                arm_ob = import_armature(armature_on, body_armature, scale_factor=SCALE_FAC) ####
                if annot_info['gender'][0][0] == 0:
                    arm_name = 'f_avg'
                else:
                    arm_name = 'm_avg'


                # set up camera mode
                # mode-A: camera is fixed
                # model is centered only for the starting frames
                # bounding box has to be recorded for later cropping
                # can be used in both non-HDR and HDR background

                # mode-B: camera is moving along its tangent line.
                # e.g, if camera is pointing to Y-axis, then it's able to move along X-axis
                # in this mode, model is alweays tracked in middle of image,
                # so the rendering area can be set to minimum
                # however, if non-HDR background is used in this mode, then the background is still
                # better to use this mode in HDR background

                # mode-C: camera is pointing to the model and tracking its movement
                # when the model rotate along 'Pelvis' vector by a certain angle
                # camera will rotate the same angle, along the same vector
                # therefore, the front camera will alweays film the front of model
                # regardless the model's movement


                for view_mode in [0, 1, 2, 3]:
                    if view_mode == 0:
                        viewpoint = [0, 0, 0]	# TODO
                    if view_mode == 1:
                        #viewpoint = [0, 0, 45]	# TODO
                        viewpoint = [0, 0, 90]	# TODO
                    if view_mode == 2:
                        #viewpoint = [0, 0, 90]	# TODO
                        viewpoint = [0, 0, -90]	# TODO
                    if view_mode == 3:
                        #viewpoint = [0, 0, -45]	# TODO
                        viewpoint = [0, 0, 180]	# TODO
                    #if view_mode == 4:
                    #    viewpoint = [0, 0, -90]	# TODO

                    if cam_mode == 0:
                        camera_mode_a(cam_ob)
                        camera_viewpoint(cam_ob, cam_parent, viewpoint)
                    if cam_mode == 1:
                        camera_mode_a(cam_ob)	# reset camera position
                        camera_viewpoint(cam_ob, cam_parent, viewpoint)
                        camera_mode_b(cam_ob, arm_ob)
                    if cam_mode == 2:
                        camera_mode_a(cam_ob)	# reset camera position
                        camera_viewpoint(cam_ob, cam_parent, viewpoint)
                        camera_mode_c(cam_ob, arm_ob)
                    if cam_mode == 3:	# background is still in this mode
                        camera_mode_a(cam_ob)	# reset camera position
                        camera_viewpoint(cam_ob, cam_parent, viewpoint)
                        #camera_mode_b(cam_ob, arm_ob)
                        #camera_mode_d(cam_ob, arm_ob)
                        cam_ob_X = cam_ob.location[0]
                        cam_ob_Y = cam_ob.location[1]
                        cam_ob_Z = cam_ob.location[2]

                    for one_fabric in fabric_patterns:
                        fab_mat = bpy.data.materials[one_fabric]
                        fab_mat_tree = fab_mat.node_tree
                        #fab_img_name = join(fabric_path, choice(fab_files))	# randomly select fabric texture
                        fab_img_name = choice(fab_files)	# randomly select fabric texture
                        fab_img = bpy.data.images.load(fab_img_name)
                        resize_fac  = np.random.normal(1.0, 0.5)
                        change_fabric_material(fab_mat_tree, fab_img, resize_fac)

                    #hdr_img_name = join(hdr_path, choice(hdr_files))
                    #hdr_img = bpy.data.images.load(hdr_img_name)
                    #hdr_resize_fac  = np.random.normal(10, 1)
                    #hdr_rotate_z = np.random.uniform(0, 6.3)
                    #if cam_mode == 1 or cam_mode == 2 or cam_mode == 3:
                    #    change_hdr_background(img=hdr_img, resize_fac=hdr_resize_fac, rotate_z=hdr_rotate_z)
                    random_hdr_background(hdr_path, hdr_files, view_mode)

                    bg_img_name = choice(nh_txt_paths)
                    bg_img = bpy.data.images.load(bg_img_name)
                    if cam_mode == 0:
                        res_paths = create_composite_nodes(bpy.context.scene.node_tree, params, img=bg_img, idx=idx)

                    bg_img_name = choice(nh_txt_paths)
                    bg_img = bpy.data.images.load(bg_img_name)
                    if cam_mode == 0:
                        res_paths = create_composite_nodes(bpy.context.scene.node_tree, params, img=bg_img, idx=idx)

                    # NOTE NOTE: make sure the save dir is what you want
                    render_output_ = join(render_output, 'run%d' % runpass, name.replace(" ", ""))
                    render_save = join(render_output_, name.replace(" ", "") + "_c%04d_%s" % (ishape+1, garment_id), 'cmode%d' % cam_mode, "view%02d/" %(view_mode))
                    # create output directory
                    if not exists(render_save):
                        mkdir_safe(render_save)
                    else:
                        print('Skip %s' % render_save)
                        continue	# skip it is it already exists

                    # batch rendering	# TODO: comment out for re-generate joint3D annotations
                    #scene.render.filepath = render_save
                    #bpy.ops.render.render(animation = True)

                    # allocate
                    dict_info = {}
                    dict_info['bg'] = np.zeros((N,), dtype=np.object) # background image path
                    dict_info['camLoc'] = np.empty((3, N), dtype='float32') # (1, 3)
                    dict_info['clipNo'] = ishape +1
                    dict_info['cloth'] = np.zeros((N,), dtype=np.object) # clothing texture image path
                    #dict_info['gender'] = np.empty(N, dtype='uint8') # 0 for male, 1 for female
                    dict_info['gender'] = annot_info['gender']
                    dict_info['joints2D'] = np.empty((2, 24, N), dtype='float32') # 2D joint positions in pixel space
                    dict_info['joints3D'] = np.empty((3, 24, N), dtype='float32') # 3D joint positions in world coordinates
                    dict_info['light'] = np.empty((4, N), dtype='float32')
                    #dict_info['pose'] = np.empty((data['poses'][0].size, N), dtype='float32') # joint angles from SMPL (CMU)
                    dict_info['pose'] = annot_info['pose'].copy()	# NOTE: important to copy and then modify
                    # TODO: to offset the camera rotation for pose parameters
                    for dict_ix in range(dict_info['pose'].shape[1]):
                        dict_info['pose'][0:3, dict_ix] = smpl_pose_offset(dict_info['pose'][0:3, dict_ix], viewpoint)

                    dict_info['sequence'] = name.replace(" ", "") + "_c%04d" % (ishape + 1)
                    #dict_info['shape'] = np.empty((ndofs, N), dtype='float32')
                    dict_info['shape'] = annot_info['shape']
                    # dict_info['zrot'] = np.empty(N, dtype='float32')
                    dict_info['camDist'] = [CAMERA_DEPTH, CAMERA_HEIGHT]
                    dict_info['stride'] = stride

                    if name.replace(" ", "").startswith('h36m'):
                        dict_info['source'] = 'h36m'
                    else:
                        dict_info['source'] = 'cmu'

                    get_real_frame = lambda ifr: ifr


                    # iterate over the keyframes and record annotations
                    # LOOP TO ANNOTATE
                    matfile_info_render = join(render_save, name.replace(" ", "") + "_c%04d_info.mat" % (ishape+1))
                    for seq_frame in range(N):
                        #scene.frame_set(get_real_frame(seq_frame))
                        scene.frame_set(get_real_frame(seq_frame + 1)) # "+1" to make rendering and annotation synchrnoized.
                        iframe = seq_frame
                        log_message("Annotating frame %d" % seq_frame)
                        dict_info['bg'][iframe] = bg_img_name
                        dict_info['cloth'][iframe] = cloth_img_name
                        dict_info['light'][:, iframe] = [lamp_0_str, lamp_1_str, lamp_2_str, lamp_3_str]
                        dict_info['camLoc'] = cam_ob.location
                        # scene.render.use_antialiasing = False
                        # scene.render.filepath = join(rgb_path, 'Image%04d.png' % get_real_frame(seq_frame))
                        # # disable render output
                        # logfile = '/dev/null'
                        # open(logfile, 'a').close()
                        # old = os.dup(1)
                        # sys.stdout.flush()
                        # os.close(1)
                        # os.open(logfile, os.O_WRONLY)
                        # # Render
                        # bpy.ops.render.render(write_still=True)
                        # # disable output redirection
                        # os.close(1)
                        # os.dup(old)
                        # os.close(old)
                        # bone locations should be saved after rendering so that the bones are updated
                        bone_locs_2D, bone_locs_3D_ = get_bone_locs(arm_name, arm_ob, scene, cam_ob)
                        # Rotate joints' 3D location, to get the correct locations relative to the camera
                        bone_locs_4D = np.concatenate((bone_locs_3D_, np.zeros((bone_locs_3D_.shape[0],1))), axis=1)
                        #bone_locs_4D_ = np.matmul(cam_parent.matrix_world, bone_locs_4D.transpose())
                        #bone_locs_4D_ = np.matmul(Matrix.Rotation(radians(180), 4, 'Z'), bone_locs_4D_)
                        # TODO: assume that camera only rotates along axis-Z, where bones' rotation is -viewpoint[2]
                        #bone_RotMat = Euler((0., 0., radians(-viewpoint[2])), 'XYZ').to_matrix().to_4x4()
                        # TODO: assume that camera rotates along Z and X sequentially
                        bone_RotMat = Euler((radians(-viewpoint[0]), 0., radians(-viewpoint[2])), 'ZYX').to_matrix().to_4x4()
                        bone_locs_4D_ = np.matmul(bone_RotMat, bone_locs_4D.transpose())
                        bone_locs_3D = bone_locs_4D_[0:3, :].transpose() / SCALE_FAC	# NOTE: Downscaling joint3D is needed!
                        dict_info['joints2D'][:, :, iframe] = np.transpose(bone_locs_2D)
                        dict_info['joints3D'][:, :, iframe] = np.transpose(bone_locs_3D)
                        #reset_loc = (bone_locs_2D.max(axis=-1) > 256).any() or (bone_locs_2D.min(axis=0) < 0).any()
                        #arm_ob.pose.bones[obname+'_root'].rotation_quaternion = Quaternion((1, 0, 0, 0))

                        if cam_mode == 3:
                            pevis_locs = bone_locs_3D_[0]
                            cam_ob.location[0] = cam_ob_X + pevis_locs[0]
                            cam_ob.location[1] = cam_ob_Y + pevis_locs[1]
                            cam_ob.location[2] = cam_ob_Z + pevis_locs[2]
                        # single rendering
                        if seq_frame+1 > PREAMBLE_FRAME_NUM-1:
                            img_file = join(render_save, "%04d.png" % (seq_frame+1))
                            scene.render.filepath = img_file
                            bpy.ops.render.render(write_still = True)

                        if frame_only_mode == True:
                            #hdr_img_name = join(hdr_path, choice(hdr_files))
                            #hdr_img = bpy.data.images.load(hdr_img_name)
                            #hdr_resize_fac  = np.random.normal(10, 1)
                            #hdr_rotate_z = np.random.uniform(0, 6.3)
                            #if cam_mode == 1 or cam_mode == 2 or cam_mode == 3:
                            #    change_hdr_background(img=hdr_img, resize_fac=hdr_resize_fac, rotate_z=hdr_rotate_z)
                            random_hdr_background(hdr_path, hdr_files, view_mode)

                            bg_img_name = choice(nh_txt_paths)
                            bg_img = bpy.data.images.load(bg_img_name)
                            if cam_mode == 0:
                                res_paths = create_composite_nodes(bpy.context.scene.node_tree, params, img=bg_img, idx=idx)

                            for one_fabric in fabric_patterns:
                                fab_mat = bpy.data.materials[one_fabric]
                                fab_mat_tree = fab_mat.node_tree
                                #fab_img_name = join(fabric_path, choice(fab_files))	# randomly select fabric texture
                                fab_img_name = choice(fab_files)	# randomly select fabric texture
                                print('FAB IMAGE NAME IS %s' %fab_img_name)
                                fab_img = bpy.data.images.load(fab_img_name)
                                resize_fac  = np.random.normal(1.0, 0.5)
                                change_fabric_material(fab_mat_tree, fab_img, resize_fac)

                    # save a .blend file for debugging:
                    # bpy.ops.wm.save_as_mainfile(filepath=join(tmp_path, 'pre.blend'))

                    # save annotation excluding png/exr data to _info.mat file
                    #import scipy.io
                    sio.savemat(matfile_info_render, dict_info, do_compression=True)

                # Initialize Blender for the next clip generation
                bpy.ops.wm.read_homefile()


if __name__ == '__main__':
    runpass = 0
    main(runpass)

