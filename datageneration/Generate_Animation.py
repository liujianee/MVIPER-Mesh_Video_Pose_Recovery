import sys
import os
import random
import math
import bpy
import numpy as np
import hashlib
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

PREAMBLE_FRAME_NUM = 10
CAMERA_DISTANCE = 8

STRIDE = 160
# TRANS_OFFSET = [0., 0.91, -0.91]

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
part_match = {'root':'root', 'bone_00':'Pelvis', 'bone_01':'L_Hip', 'bone_02':'R_Hip',
              'bone_03':'Spine1', 'bone_04':'L_Knee', 'bone_05':'R_Knee', 'bone_06':'Spine2',
              'bone_07':'L_Ankle', 'bone_08':'R_Ankle', 'bone_09':'Spine3', 'bone_10':'L_Foot',
              'bone_11':'R_Foot', 'bone_12':'Neck', 'bone_13':'L_Collar', 'bone_14':'R_Collar',
              'bone_15':'Head', 'bone_16':'L_Shoulder', 'bone_17':'R_Shoulder', 'bone_18':'L_Elbow',
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

# create the different passes that we render
def create_composite_nodes(tree, params, img=None, idx=0):
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

# computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
    return(cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)

def init_scene(scene, params, gender='female'):
    # load fbx model
    bpy.ops.import_scene.fbx(filepath=join(params['smpl_data_folder'], 'basicModel_%s_lbs_10_207_0_v1.0.2.fbx' % gender[0]),
                             axis_forward='Y', axis_up='Z', global_scale=100)
    obname = '%s_avg' % gender[0]
    ob = bpy.data.objects[obname]
    ob.data.use_auto_smooth = False  # autosmooth creates artifacts
    # assign the existing spherical harmonics material
    ob.active_material = bpy.data.materials['Material']
    # delete the default cube (which held the material)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete(use_global=False)
    # set camera properties and initial position
    bpy.ops.object.select_all(action='DESELECT')
    cam_ob = bpy.data.objects['Camera']
    scn = bpy.context.scene
    scn.objects.active = cam_ob
    #cam_ob.matrix_world = Matrix(((0., 0., 1, params['camera_distance']),(0., -1, 0., -1.0),(-1., 0., 0., 0.),(0.0, 0.0, 0.0, 1.0)))
    #cam_ob.matrix_world = Matrix(((1., 0., 0, 0),(0., 0, -1., -8.0),(0., 1., 0., 1.),(0.0, 0.0, 0.0, 1.0)))
    cam_ob.matrix_world = Matrix(((1., 0., 0, 0),(0., 0, -1., -CAMERA_DISTANCE),(0., 1., 0., 1.),(0.0, 0.0, 0.0, 1.0)))
    cam_ob.data.angle = math.radians(40)	# this is field of view
    cam_ob.data.lens =  60
    cam_ob.data.clip_start = 0.1
    cam_ob.data.sensor_width = 32
    # setup an empty object in the center which will be the parent of the Camera
    # this allows to easily rotate an object around the origin
    scn.cycles.film_transparent = True
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
    ob.data.shape_keys.animation_data_clear()
    arm_ob = bpy.data.objects['Armature']
    arm_ob.animation_data_clear()
    return(ob, obname, arm_ob, cam_ob)

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


def get_bone_locs(obname, arm_ob, scene, cam_ob):
    n_bones = 24
    render_scale = scene.render.resolution_percentage / 100
    render_size = (int(scene.render.resolution_x * render_scale),
                   int(scene.render.resolution_y * render_scale))
    bone_locations_2d = np.empty((n_bones, 2))
    bone_locations_3d = np.empty((n_bones, 3), dtype='float32')
    # obtain the coordinates of each bone head in image space
    for ibone in range(n_bones):
        bone = arm_ob.pose.bones[obname+'_'+part_match['bone_%02d' % ibone]]
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

import time
start_time = 0.
def log_message(message):
    elapsed_time = time.time() - start_time
    print("[%.2f s] %s" % (elapsed_time, message))

def main(runpass=None, idx=None, idx_info=None, stride=None, cmu_idx=0):
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

    #if runpass == None:
    #    log_message("WARNING: runpass not specified, using default value 0")
    #    runpass = 0

    #if idx == None:
    #    log_message("WARNING: idx not specified, using default value 0")
    #    idx = 0

    #if stride == None:
    #    log_message("WARNING: stride not specified, using default value 50")
    #    stride = 50

    log_message("input idx: %d" % idx)
    log_message("input stride: %d" % stride)

    # import idx info (name, split)
    #idx_info = load(open("pkl/idx_info.pickle", 'rb'))

    # get runpass
    #(runpass, idx) = divmod(idx, len(idx_info))

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

    # compute number of cuts
    #nb_ishape = max(1, int(np.ceil((idx_info['nb_frames'] - (clipsize - stride))/stride)))
    nb_ishape = 1
    log_message("Max ishape: %d" % (nb_ishape - 1))

    #if ishape == None:
    #    exit(1)

    #assert(ishape < nb_ishape)

    # name is set given idx
    name = idx_info['name']
    #output_path = join(output_path, 'run%d' % runpass, name.replace(" ", ""))
    #params['output_path'] = output_path

    #tmp_path = join(tmp_path, 'run%d_%s_c%04d' % (runpass, name.replace(" ", ""), (ishape + 1)))
    #params['tmp_path'] = tmp_path

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
    #import hashlib
    #s = "synth_data:%d:%d:%d" % (idx, runpass, ishape)
    #seed_number = int(hashlib.sha1(s.encode('utf-8')).hexdigest(), 16) % (10 ** 8)
    #log_message("GENERATED SEED %d from string '%s'" % (seed_number, s))
    #random.seed(seed_number)
    #np.random.seed(seed_number)

    #if(output_types['vblur']):
    #    vblur_factor = np.random.normal(0.5, 0.5)
    #    params['vblur_factor'] = vblur_factor

    #log_message("Setup Blender")

    # create copy-spher.harm. directory if not exists
    #sh_dir = join(tmp_path, 'spher_harm')
    #if not exists(sh_dir):
    #    mkdir_safe(sh_dir)

    #sh_dst = join(sh_dir, 'sh_%02d_%05d.osl' % (runpass, idx))
    #os.system('cp spher_harm/sh.osl %s' % sh_dst)

    genders = {0: 'female', 1: 'male'}
    # pick random gender
    #gender = choice(genders)

    for ishape in range(nb_ishape):
    #for ishape in range(1):

        s = "synth_data:%d:%d:%d" % (idx, runpass, ishape)
        seed_number = int(hashlib.sha1(s.encode('utf-8')).hexdigest(), 16) % (10 ** 8)
        log_message("GENERATED SEED %d from string '%s'" % (seed_number, s))

        random.seed(seed_number)
        gender = choice(genders)
        #gender = 'male'

        output_path = join(output_path, 'run%d' % runpass, name.replace(" ", "") + "_%s" % (gender))
        params['output_path'] = output_path

        log_message("current ishape: %d" % ishape)
        scene = bpy.data.scenes['Scene']
        scene.render.engine = 'CYCLES'
        bpy.data.materials['Material'].use_nodes = True
        scene.cycles.shading_system = True
        scene.use_nodes = True

        #log_message("Listing background images")
        #bg_names = join(bg_path, '%s_img.txt' % idx_info['use_split'])
        #nh_txt_paths = []
        #with open(bg_names) as f:
        #    for line in f:
        #        nh_txt_paths.append(join(bg_path, line))

        # grab clothing names
        #log_message("clothing: %s" % clothing_option)
        #with open( join(smpl_data_folder, 'textures', '%s_%s.txt' % ( gender, idx_info['use_split'] ) ) ) as f:
        #    txt_paths = f.read().splitlines()

        # if using only one source of clothing
        #if clothing_option == 'nongrey':
        #    txt_paths = [k for k in txt_paths if 'nongrey' in k]
        #elif clothing_option == 'grey':
        #    txt_paths = [k for k in txt_paths if 'nongrey' not in k]

        # random clothing texture
        #cloth_img_name = choice(txt_paths)
        #cloth_img_name = join(smpl_data_folder, cloth_img_name)
        #cloth_img = bpy.data.images.load(cloth_img_name)

        # random background
        #bg_img_name = choice(nh_txt_paths)[:-1]
        #bg_img = bpy.data.images.load(bg_img_name)

        log_message("Loading parts segmentation")
        beta_stds = np.load(join(smpl_data_folder, ('%s_beta_stds.npy' % gender)))

        log_message("Building materials tree")
        # mat_tree = bpy.data.materials['Material'].node_tree
        # create_sh_material(mat_tree, sh_dst, cloth_img)
        # res_paths = create_composite_nodes(scene.node_tree, params, img=bg_img, idx=idx)	# Only human model is saved in this step, no background

        log_message("Loading smpl data")
        smpl_data = np.load(join(smpl_data_folder, smpl_data_filename))

        log_message("Initializing scene")
        #camera_distance = np.random.normal(8.0, 1)
        camera_distance = CAMERA_DISTANCE
        params['camera_distance'] = camera_distance ###############################################################
        ob, obname, arm_ob, cam_ob = init_scene(scene, params, gender)

        setState0()
        ob.select = True
        bpy.context.scene.objects.active = ob
        segmented_materials = False #True: 0-24, False: expected to have 0-1 bg/fg

        log_message("Creating materials segmentation")

        # create material segmentation
        if segmented_materials:
            materials = create_segmentation(ob, params)
            prob_dressed = {'leftLeg':.5, 'leftArm':.9, 'leftHandIndex1':.01,
                            'rightShoulder':.8, 'rightHand':.01, 'neck':.01,
                            'rightToeBase':.9, 'leftShoulder':.8, 'leftToeBase':.9,
                            'rightForeArm':.5, 'leftHand':.01, 'spine':.9,
                            'leftFoot':.9, 'leftUpLeg':.9, 'rightUpLeg':.9,
                            'rightFoot':.9, 'head':.01, 'leftForeArm':.5,
                            'rightArm':.5, 'spine1':.9, 'hips':.9,
                            'rightHandIndex1':.01, 'spine2':.9, 'rightLeg':.5}
        else:
            materials = {'FullBody': bpy.data.materials['Material']}
            prob_dressed = {'FullBody': .6}

        # orig_pelvis_loc = (arm_ob.matrix_world.copy() * arm_ob.pose.bones[obname+'_Pelvis'].head.copy()) - Vector((-1., 1., 1.))
        orig_cam_loc = cam_ob.location.copy()

        # unblocking both the pose and the blendshape limits
        for k in ob.data.shape_keys.key_blocks.keys():
            bpy.data.shape_keys["Key"].key_blocks[k].slider_min = -10
            bpy.data.shape_keys["Key"].key_blocks[k].slider_max = 10

        log_message("Loading body data")
        cmu_parms, fshapes, name = load_body_data(smpl_data, ob, obname, idx=cmu_idx, gender=gender)

        log_message("Loaded body data for %s" % name)

        nb_fshapes = len(fshapes)
        if idx_info['use_split'] == 'train':
            fshapes = fshapes[:int(nb_fshapes*0.8)]
        elif idx_info['use_split'] == 'test':
            fshapes = fshapes[int(nb_fshapes*0.8):]

        # pick random real body shape
        random.seed(seed_number)
        shape = choice(fshapes) #+random_shape(.5) can add noise

        # example shapes
        #shape = np.zeros(10) #average
        #shape = np.array([ 2.25176191, -3.7883464 ,  0.46747496,  3.89178988,  2.20098416,  0.26102114, -3.07428093,  0.55708514, -3.94442258, -2.88552087]) #fat

        ndofs = 10

        scene.objects.active = arm_ob
        orig_trans = np.asarray(arm_ob.pose.bones[obname+'_Pelvis'].location).copy()

        # create output directory
        if not exists(output_path):
            mkdir_safe(output_path)

        # spherical harmonics material needs a script to be loaded and compiled
        #  scs = []
        #  for mname, material in materials.items():
        #      scs.append(material.node_tree.nodes['Script'])
        #      scs[-1].filepath = sh_dst
        #      scs[-1].update()

        #rgb_dirname = name.replace(" ", "") + '_c%04d.mp4' % (ishape + 1)
        #rgb_path = join(tmp_path, rgb_dirname)

        data = cmu_parms[name]

        #fbegin = ishape*stepsize*stride
        #fend = min(ishape*stepsize*stride + stepsize*clipsize, len(data['poses']))
        fbegin = 0
        fend = len(data['poses'])

        log_message("Computing how many frames to allocate")
        #N = len(data['poses'][fbegin:fend:stepsize])
        N = len(data['poses'][fbegin:fend:stepsize]) + PREAMBLE_FRAME_NUM
        log_message("Allocating %d frames in mat file" % N)

        # force recomputation of joint angles unless shape is all zeros
        curr_shape = np.zeros_like(shape)
        nframes = len(data['poses'][::stepsize])

        matfile_info = join(output_path, name.replace(" ", "") + "_c%04d_info.mat" % (ishape+1))
        log_message('Working on %s' % matfile_info)

        # allocate
        dict_info = {}
        # dict_info['bg'] = np.zeros((N,), dtype=np.object) # background image path
        dict_info['camLoc'] = np.empty((3, N), dtype='float32') # (1, 3)
        dict_info['clipNo'] = ishape +1
        # dict_info['cloth'] = np.zeros((N,), dtype=np.object) # clothing texture image path
        dict_info['gender'] = np.empty(N, dtype='uint8') # 0 for male, 1 for female
        dict_info['joints2D'] = np.empty((2, 24, N), dtype='float32') # 2D joint positions in pixel space
        dict_info['joints3D'] = np.empty((3, 24, N), dtype='float32') # 3D joint positions in world coordinates
        # dict_info['light'] = np.empty((9, N), dtype='float32')
        dict_info['pose'] = np.empty((data['poses'][0].size, N), dtype='float32') # joint angles from SMPL (CMU)
        dict_info['sequence'] = name.replace(" ", "") + "_c%04d" % (ishape + 1)
        dict_info['shape'] = np.empty((ndofs, N), dtype='float32')
        # dict_info['zrot'] = np.empty(N, dtype='float32')
        dict_info['camDist'] = camera_distance
        dict_info['stride'] = stride
        dict_info['seednumber'] = seed_number

        if name.replace(" ", "").startswith('h36m'):
            dict_info['source'] = 'h36m'
        else:
            dict_info['source'] = 'cmu'

        #if(output_types['vblur']):
        #    dict_info['vblur_factor'] = np.empty(N, dtype='float32') #############################################

        # for each clipsize'th frame in the sequence
        get_real_frame = lambda ifr: ifr
        #random_zrot = 0
        reset_loc = False
        batch_it = 0
        curr_shape = reset_joint_positions(orig_trans, shape, ob, arm_ob, obname, scene,
                                           cam_ob, smpl_data['regression_verts'], smpl_data['joint_regressor'])
        #random_zrot = 2*np.pi*np.random.rand()

        arm_ob.animation_data_clear()
        cam_ob.animation_data_clear() #####################################################################################################

        # create a keyframe animation with pose, translation, blendshapes and camera motion
        # LOOP TO CREATE 3D ANIMATION

        # rotate model to front up-right
        #arm_ob.pose.bones[obname+'_root'].rotation_quaternion = Quaternion(Euler((math.radians(-90), 0, 0), 'XYZ'))
        #JIANL: Weired! Weired! Weired!
        #JIANL: The value is different for system-terminal and blender's python console (if the rotation was applied to "root")
        #JIANL: To solve this issue, modify arm_ob.matrix_world instead!
        # This is to make local coordinate the same as global coordinate
        arm_ob.matrix_world = Matrix(((1.0000, 0.0000, 0.0000, 0.0000),
                                      (0.0000, 1.0000, 0.0000, 0.0000),
                                      (0.0000, 0.0000, 1.0000, 0.0000),
                                      (0.0000, 0.0000, 0.0000, 1.0000)))

        curr_pelvis_loc = arm_ob.matrix_world.copy() * arm_ob.pose.bones[obname+'_Pelvis'].head.copy()

        # After resetting the local coordinate, curr_pelvis_loc[1] is y-axis value,
        # curr_pelvis_loc[2] is the z-axis value,
        # [math.radians(90), 0, 0] rotation to get T-pose
        # so, move curr_pelvis_loc[1] along y-axis (negative)
        # and move (curr_pelvis_loc[1]-curr_pelvis_loc[2]) along z-axis (positive)
        # this will result in a right pelvis position, which makes foot touching the ground
        TRANS_OFFSET = [0., curr_pelvis_loc[1], -(curr_pelvis_loc[1]-curr_pelvis_loc[2])] # OFFSET to make model stand on ground at origin
        #print(TRANS_OFFSET)

        # interpolate frames to add T-pose transition
        for seq_frame in range(PREAMBLE_FRAME_NUM):
            iframe = seq_frame
            scene.frame_set(get_real_frame(seq_frame))
            #trans = data['trans'][fbegin:fend:stepsize][0].copy()
            #trans = np.zeros(3) - [0., 0.91, 0.]
            trans = np.zeros(3) - TRANS_OFFSET	# make model stand on ground at origin
            pose = data['poses'][fbegin:fend:stepsize][0].copy()
            pose[0] = math.radians(90) + seq_frame * ((pose[0]-math.radians(90))/PREAMBLE_FRAME_NUM)
            pose[1] = seq_frame * (pose[1]/PREAMBLE_FRAME_NUM)		# interpolate rotation
            pose[2] = seq_frame * (pose[2]/PREAMBLE_FRAME_NUM)		# interpolate rotation
            pose[3:] = seq_frame * (pose[3:]/PREAMBLE_FRAME_NUM)	# interpolate rotation
            apply_trans_pose_shape(Vector(trans), pose, shape, ob, arm_ob, obname, scene, cam_ob, get_real_frame(seq_frame))
            dict_info['shape'][:, iframe] = shape[:ndofs]
            dict_info['pose'][:, iframe] = pose
            dict_info['gender'][iframe] = list(genders)[list(genders.values()).index(gender)]
            #if(output_types['vblur']):
            #    dict_info['vblur_factor'][iframe] = vblur_factor
            #arm_ob.pose.bones[obname+'_root'].rotation_quaternion = Quaternion(Euler((0, 0, random_zrot), 'XYZ'))
            arm_ob.pose.bones[obname+'_root'].keyframe_insert('rotation_quaternion', frame=get_real_frame(seq_frame))
            #dict_info['zrot'][iframe] = random_zrot
            scene.update()
            new_pelvis_loc = arm_ob.matrix_world.copy() * arm_ob.pose.bones[obname+'_Pelvis'].head.copy()
            # keep camera tracking the model on a plane, modify if axis locking is required
            # camera location is recorded for rendering
            dict_info['camLoc'][:, iframe] = orig_cam_loc.copy() + new_pelvis_loc - Vector([0, 0, 1]) #####################################################

        for seq_frame_, (pose_, trans_) in enumerate(zip(data['poses'][fbegin:fend:stepsize], data['trans'][fbegin:fend:stepsize])):
            seq_frame = seq_frame_ + PREAMBLE_FRAME_NUM
            iframe = seq_frame
            scene.frame_set(get_real_frame(seq_frame))
            # apply the translation, pose and shape to the character
            pose = pose_.copy()
            #trans = trans_.copy() - data['trans'][fbegin:fend:stepsize][0] - [0., 0.91, 0.]
            trans = trans_.copy() - data['trans'][fbegin:fend:stepsize][0] - TRANS_OFFSET # make model stand on ground at origin
            #pose[0:3] = 0
            #trans[:] = trans[[0,2,1]]
            #print(trans)
            apply_trans_pose_shape(Vector(trans), pose, shape, ob, arm_ob, obname, scene, cam_ob, get_real_frame(seq_frame))
            dict_info['shape'][:, iframe] = shape[:ndofs]
            dict_info['pose'][:, iframe] = pose
            dict_info['gender'][iframe] = list(genders)[list(genders.values()).index(gender)]
            #if(output_types['vblur']):
            #    dict_info['vblur_factor'][iframe] = vblur_factor
            #arm_ob.pose.bones[obname+'_root'].rotation_quaternion = Quaternion(Euler((0, 0, random_zrot), 'XYZ'))
            arm_ob.pose.bones[obname+'_root'].keyframe_insert('rotation_quaternion', frame=get_real_frame(seq_frame))
            #dict_info['zrot'][iframe] = random_zrot
            scene.update()
            #print(seq_frame)
            new_pelvis_loc = arm_ob.matrix_world.copy() * arm_ob.pose.bones[obname+'_Pelvis'].head.copy()
            dict_info['camLoc'][:, iframe] = orig_cam_loc.copy() + new_pelvis_loc - Vector([0, 0, 1]) #####################################################

        matfile_info = join(output_path, name.replace(" ", "") + "_c%04d_info.mat" % (ishape+1))
        # SAVE DATASET
        # save .obj for human model
        scene.frame_set(get_real_frame(0))
        obname = '%s_avg' % gender[0]
        ob = bpy.data.objects[obname]
        scn = bpy.context.scene
        scn.objects.active = ob
        #obj_filename = name + '_' + gender + '.obj'
        obj_body_saved = join(output_path, name.replace(" ", "") + "_c%04d_%s.obj" % (ishape+1, gender))
        bpy.ops.export_scene.obj(filepath=obj_body_saved, check_existing=True, axis_forward='-Z', axis_up='Y', filter_glob="*.obj;*.mtl", use_selection=True, use_animation=False, use_mesh_modifiers=True, use_edges=True, use_smooth_groups=False, use_smooth_groups_bitflags=False, use_normals=True, use_uvs=True, use_materials=False, use_triangles=False, use_nurbs=False, use_vertex_groups=False, use_blen_objects=True, group_by_object=False, group_by_material=False, keep_vertex_order=False, global_scale=1, path_mode='AUTO')

        # save .mdd for animation
        #mdd_filename = name + '_' + gender + '.mdd'
        mdd_body_saved = join(output_path, name.replace(" ", "") + "_c%04d_%s.mdd" % (ishape+1, gender))
        # to align .mdd to .fbx, frame_start is set to 0, and use_rest_frame is true
        bpy.ops.export_shape.mdd(filepath=mdd_body_saved, check_existing=True, filter_glob="*.mdd", fps=25, frame_start=0, frame_end=N, use_rest_frame=True)

        # save .fbx for animation
        #fbx_filename = name + '_' + gender + '.fbx'
        fbx_body_saved = join(output_path, name.replace(" ", "") + "_c%04d_%s.fbx" % (ishape+1, gender))
        # export "ARMATURE" only to save space
        bpy.ops.export_scene.fbx(filepath=fbx_body_saved, check_existing=True, axis_forward='-Z', axis_up='Y', filter_glob="*.fbx", version='BIN7400', ui_tab='MAIN', use_selection=False, global_scale=1, apply_unit_scale=True, bake_space_transform=False, object_types={'ARMATURE'}, use_mesh_modifiers=False, mesh_smooth_type='OFF', use_mesh_edges=False, use_tspace=False, use_custom_props=False, add_leaf_bones=True, primary_bone_axis='Y', secondary_bone_axis='X', use_armature_deform_only=False, armature_nodetype='NULL', bake_anim=True, bake_anim_use_all_bones=True, bake_anim_use_nla_strips=True, bake_anim_use_all_actions=True, bake_anim_force_startend_keying=True, bake_anim_step=1, bake_anim_simplify_factor=1, use_anim=True, use_anim_action_all=True, use_default_take=True, use_anim_optimize=True, anim_optimize_precision=6, path_mode='AUTO', embed_textures=False, batch_mode='OFF', use_batch_own_dir=True, use_metadata=True)

        # iterate over the keyframes and record annotations
        # LOOP TO ANNOTATE
        #for seq_frame, (pose, trans) in enumerate(zip(data['poses'][fbegin:fend:stepsize], data['trans'][fbegin:fend:stepsize])):
        for seq_frame in range(N):
            scene.frame_set(get_real_frame(seq_frame))
            iframe = seq_frame
            # dict_info['bg'][iframe] = bg_img_name
            # dict_info['cloth'][iframe] = cloth_img_name
            # dict_info['light'][:, iframe] = sh_coeffs
            # scene.render.use_antialiasing = False
            # scene.render.filepath = join(rgb_path, 'Image%04d.png' % get_real_frame(seq_frame))
            log_message("Annotating frame %d" % seq_frame)
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
            bone_locs_2D, bone_locs_3D = get_bone_locs(obname, arm_ob, scene, cam_ob)
            dict_info['joints2D'][:, :, iframe] = np.transpose(bone_locs_2D)
            dict_info['joints3D'][:, :, iframe] = np.transpose(bone_locs_3D)
            #reset_loc = (bone_locs_2D.max(axis=-1) > 256).any() or (bone_locs_2D.min(axis=0) < 0).any()
            #arm_ob.pose.bones[obname+'_root'].rotation_quaternion = Quaternion((1, 0, 0, 0))

        # save a .blend file for debugging:
        # bpy.ops.wm.save_as_mainfile(filepath=join(tmp_path, 'pre.blend'))

        # save annotation excluding png/exr data to _info.mat file
        import scipy.io
        scipy.io.savemat(matfile_info, dict_info, do_compression=True)

        # Initialize Blender for the next clip generation
        bpy.ops.wm.read_homefile()


if __name__ == '__main__':

    idx_info = load(open("pkl/idx_info.pickle", 'rb'))

    whole_info_list = [idx_info[i]['name'] for i in range(len(idx_info))]
    cmu_info_list = [item for item in whole_info_list if 'h36m' not in item]
    h36m_info_list = [item for item in whole_info_list if 'h36m' in item]

    cmu_selected = './animate_out/action_list_1.txt'
    with open(cmu_selected, 'r') as f:
        lines = list(filter(None, f.read().split('\n')))
    cmu_selected_list = [item.split(',')[0] for item in lines]


    # NOTE: this is saved as run0
    for idx in range(2):
        (runpass, idx) = divmod(idx, len(idx_info))
        main(runpass=runpass, idx=idx, idx_info=idx_info, stride=STRIDE)


    # NOTE: this is saved as run1
    #for cmu_name in cmu_selected_list:
    #    if cmu_name in whole_info_list:
    #        idx = whole_info_list.index(cmu_name)
    #        cmu_idx = idx
    #        main(runpass=runpass, idx=idx, idx_info=idx_info, stride=STRIDE, cmu_idx=cmu_idx)

