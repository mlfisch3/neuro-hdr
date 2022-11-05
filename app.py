from neuro_hdr import bimef, entropy, xentropy, KL, autoscale_array, array_info, log_memory
from neuro_hdr import joint_entropy, mutual_information, variation_of_information, normalized_variation_of_information, conditional_entropy
import cv2
import numpy as np
from matplotlib import image as img
import streamlit as st
import datetime
from psutil import Process
import os
from os import getpid
import subprocess

def timestamp():
    return datetime.datetime.now().isoformat() #strftime("%Y%m%d_%H%M%S")


st.set_page_config(page_title="Neuro HDR", layout="wide")

def initialize_session():

    if 'query_params' not in st.session_state:
        st.session_state.query_params = {}
        st.session_state.query_params['console'] = False

    if 'show_console' not in st.session_state:
        st.session_state.show_console = False

    if 'console_out' not in st.session_state:
        st.session_state.console_out = ''

    if 'console_in' not in st.session_state:
        st.session_state.console_in = ''


def run_command():
    print(f'[{timestamp()}] st.session_state.console_in: {st.session_state.console_in}')
    try:
        st.session_state.console_out = str(subprocess.check_output(st.session_state.console_in, shell=True, text=True))
        st.session_state.console_out_timestamp = f'{timestamp()}'
    except subprocess.CalledProcessError as e:
        st.session_state.console_out = f'exited with error\nreturncode: {e.returncode}\ncmd: {e.cmd}\noutput: {e.output}\nstderr: {e.stderr}'
        st.session_state.console_out_timestamp = f'{timestamp()}'

    #print(f'[{timestamp()}] st.session_state.console_out: {st.session_state.console_out}')


def run_app(default_granularity=0.1, default_speed=10, default_power=0.8, default_smoothness=0.3, 
            default_dim_size=(50), default_dim_threshold=0.5, default_a=-0.3293, default_b=1.1258, default_exposure_ratio=-1):

    log_memory('run_app||B')
    container = st.sidebar.container()
    with container:    
        pid = getpid()
        placeholder = st.empty()
        if st.session_state.show_console:
            with placeholder.container():
                with st.expander("console", expanded=True):
                    with st.form('console'):
                        command = st.text_input(f'[{pid}] {timestamp()}', str(st.session_state.console_in), key="console_in")
                        submitted = st.form_submit_button('run', help="coming soon", on_click=run_command)

                        st.write(f'IN: {command}')
                        st.text(f'OUT:\n{st.session_state.console_out}')
                    file_name = st.text_input("File Name", "")
                    if os.path.isfile(file_name):
                        button = st.download_button(label="Download File", data=Path(file_name).read_bytes(), file_name=file_name, key="console_download")
        else:
             placeholder.empty()
    #@st.cache(max_entries=1, show_spinner=False)
    def adjust_intensity(
                         array, 
                         exposure_ratio=-1, enhance=0.8, 
                         a=-0.3293, b=1.1258, lamda=0.3, 
                         sigma=5, scale=0.1, sharpness=0.001, 
                         dim_threshold=0.5, dim_size=(50,50), 
                         solver='cg', CG_prec='ILU', CG_TOL=0.1, LU_TOL=0.015, MAX_ITER=50, FILL=50, 
                         lo=1, hi=7, npoints=20
                         ):
        

        return bimef(
                         array[:,:,[2,1,0]], 
                         exposure_ratio=exposure_ratio, enhance=enhance, 
                         a=a, b=b, lamda=lamda, 
                         sigma=sigma, scale=scale, sharpness=sharpness, 
                         dim_threshold=dim_threshold, dim_size=dim_size, 
                         solver=solver, CG_prec='ILU', CG_TOL=CG_TOL, LU_TOL=LU_TOL, MAX_ITER=MAX_ITER, FILL=FILL, 
                         lo=lo, hi=hi, npoints=npoints
                         ) 


    # st.sidebar.markdown("##### A faster instance of this app is running on Streamlit Cloud #####")
    # href = f'<a href="https://share.streamlit.io/mlfisch3/luminon/main/app.py">Streamlit Cloud instance</a>'
    # st.sidebar.markdown(href, unsafe_allow_html=True)

    log_memory('run_app|file_uploader|B')
    fImage = st.sidebar.file_uploader("Upload image file:")
    log_memory('run_app|file_uploader|E')

    speed = float(st.sidebar.text_input('Speed   (default = 10)', str(default_speed)))
    if (speed < 1) or (speed > 10):
        granularity = default_granularity
    else:
        granularity = 1.0 / speed
    power = float(st.sidebar.text_input('Power     (default = 0.8)', str(default_power)))
    smoothness = float(st.sidebar.text_input('Smoothness   (default = 0.3)', str(default_smoothness)))
    #exposure_sample = int(st.sidebar.text_input('Sample   (default = 50)', str(default_dim_size)))
    #sensitivity = float(st.sidebar.text_input('Sensitivity   (default = 0.5)', str(default_dim_threshold)))
    a = float(st.sidebar.text_input('Camera A   (default = -0.3293)', str(default_a)))
    b = float(st.sidebar.text_input('Camera B   (default = 1.1258)', str(default_b)))
    exposure_ratio = float(st.sidebar.text_input('Exposure Ratio   (default = -1 (auto))', str(default_exposure_ratio)))

    col1, col2 = st.columns(2)

    if fImage is not None:
        
        input_file_name = str(fImage.__dict__['name'])
        input_file_ext = '.' + str(input_file_name.split('.')[-1])
        input_file_basename = input_file_name.replace(input_file_ext, '')
        log_memory('run_app|np.frombuffer|B')
        np_array = np.frombuffer(fImage.getvalue(), np.uint8)
        log_memory('run_app|np.frombuffer|E')
        log_memory('run_app|cv2.imdecode|B')
        image_np = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        log_memory('run_app|cv2.imdecode|E')

        with col1:        

            st.header(f'Original Image')
            log_memory('run_app|st.image|B')
            st.image(image_np[:,:,[2,1,0]])
            log_memory('run_app|st.image|E')

            input_file_name = st.text_input('Download Original Image As', input_file_name)
            ext = '.' + input_file_name.split('.')[-1]
            log_memory('run_app|cv2.imencode|B')
            image_np_binary = cv2.imencode(ext, image_np)[1].tobytes()
            log_memory('run_app|cv2.imencode|E')

            button = st.download_button(label = "Download Original Image", data = image_np_binary, file_name = input_file_name, mime = "image/png")

        start = datetime.datetime.now()
        log_memory('run_app|adjust_intensity|B')
        image_np_ai = adjust_intensity(image_np, exposure_ratio=exposure_ratio, scale=granularity, enhance=power, lamda=smoothness, a=a, b=b)
        log_memory('run_app|adjust_intensity|E')

        end = datetime.datetime.now()
        process_time = (end - start).total_seconds()
        print(f'[{datetime.datetime.now().isoformat()}]  Processing time: {process_time:.5f} s')

        processed_file_name = input_file_basename + '_AI' + input_file_ext
        with col2:        
            st.header(f'Enhanced Image')
            log_memory('run_app|st.image|B')            
            st.image(image_np_ai, clamp=True)
            log_memory('run_app|st.image|E')

            output_file_name = st.text_input('Download Enhanced Image As', processed_file_name)
            ext = '.' + output_file_name.split('.')[-1]
            log_memory('run_app|cv2.imencode|B')
            image_np_ai_binary = cv2.imencode(ext, image_np_ai[:,:,[2,1,0]])[1].tobytes()
            log_memory('run_app|cv2.imencode|E')

            button = st.download_button(label = "Download Enhanced Image", data = image_np_ai_binary, file_name = output_file_name, mime = "image/png")

        st.text('\n\n\n\n\n\n\n\n')
        st.text('*Supported file extensions: jpg, jpeg, png, gif, bmp, pdf, svg, eps')
            
        log_memory('run_app|array_info|B')
        image_np_info, image_np_info_str = array_info(image_np, print_info=False, return_info=True, return_info_str=True)
        log_memory('run_app|array_info|E')
        log_memory('run_app|array_info|B')
        image_np_ai_info, image_np_ai_info_str = array_info(image_np_ai, print_info=False, return_info=True, return_info_str=True)
        log_memory('run_app|array_info|E')

        log_memory('run_app|xentropy|B')
        relative_entropy = xentropy(image_np, image_np_ai)
        log_memory('run_app|xentropy|E')
        log_memory('run_app|kl_divergence|B')
        kl_divergence = KL(image_np, image_np_ai)
        log_memory('run_app|kl_divergence|E')
        # log_memory('run_app|joint_entropy|B')
        # s_joint = joint_entropy(image_np, image_np_ai)
        # log_memory('run_app|joint_entropy|E')
        # log_memory('run_app|mutual_information|B')
        # i_mutual = mutual_information(image_np, image_np_ai)
        # log_memory('run_app|mutual_information|E')
        # log_memory('run_app|variation_of_information|B')
        # voi = variation_of_information(image_np, image_np_ai)
        # log_memory('run_app|variation_of_information|E')
        # log_memory('run_app|normalized_variation_of_information|B')
        # nvoi = normalized_variation_of_information(image_np, image_np_ai)
        # log_memory('run_app|normalized_variation_of_information|E')

        entropy_change_abs = image_np_ai_info['entropy'] - image_np_info['entropy']
        entropy_change_rel = (image_np_ai_info['entropy'] / image_np_info['entropy']) - 1.0

        log_memory('run_app|show statistics|B')
        st.sidebar.text(f'entropy change: {entropy_change_abs:.4f} ({entropy_change_rel * 100.0:.4f} %)\n')        
        st.sidebar.text(f'relative entropy: {relative_entropy:.4f}')
        st.sidebar.text(f'KL divergence: {kl_divergence:.4f}')
        # st.sidebar.text(f'joint entropy: {s_joint:.4f}')
        # st.sidebar.text(f'mutual information: {i_mutual:.4f}')
        # st.sidebar.text(f'variation of information (VOI): {voi:.4f}')
        # st.sidebar.text(f'normalized VOI: {nvoi:.4f}\n')
        
        st.sidebar.text("Pixel Statistics [Original Image]:")
        
        st.sidebar.text(image_np_info_str)
        
        st.sidebar.text("\n\n\n\n\n")
        
        st.sidebar.text("Pixel Statistics [Enhanced Image]:")

        st.sidebar.text(image_np_ai_info_str)
        log_memory('run_app|show_statistics|E')
        log_memory('run_app||E')

        st.write(st.session_state)

if __name__ == '__main__':
    
    log_memory('main|run_app|B')
    initialize_session()
    ss = st.session_state

    query_params = st.experimental_get_query_params()
    for k,v in query_params.items():
        ss.query_params[k] = v[0]
        ss.query_params.setdefault(k,v[0])

    if 'console' in query_params:
        st.session_state.show_console = query_params['console'][0]
    else:
        st.session_state.show_console = False

    run_app()

    log_memory('main|run_app|E')
