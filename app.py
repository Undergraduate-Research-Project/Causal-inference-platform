from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session, jsonify, send_file
import os
import pandas as pd
import numpy as np
import logging
import networkx as nx 
import subprocess
import json
from datetime import datetime
from column_type_detector import detect_column_types as detect_analysis_types 
from detect_type import detect_column_types  
from EI.calculator import *
from api.deepseek import DeepSeekClient  

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
client = DeepSeekClient()  
selected_var = ""


if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists('static'):
    os.makedirs('static')

@app.route('/')
def log_in():
    return render_template('log-in.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/data-upload.html', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify(success=False, message='没有文件被上传'), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify(success=False, message='没有选择文件'), 400
        
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            try:
                file.save(filepath)
            except PermissionError as e:
                return jsonify(success=False, message=f"权限错误: {e}"), 500
            session['uploaded_filename'] = filepath
            session['preparation_filepath'] = filepath
            return jsonify(success=True, message='文件上传成功', redirect_url=url_for('data_preparation'))

    return render_template('data-upload.html')

@app.route('/api/check-upload-status', methods=['GET'])
def check_upload_status():
    file_uploaded = 'uploaded_filename' in session
    return jsonify(fileUploaded=file_uploaded)

@app.route('/api/upload-history', methods=['GET'])
def upload_history():
    files = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(file_path):
            upload_time = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %I:%M %p')
            files.append({'filename': filename, 'upload_time': upload_time})
    return jsonify(files)

@app.route('/api/plot-data', methods=['GET'])
def plot_data():
    # 获取前端传来的列名
    column = request.args.get('column')
    # 获取会话中的文件路径
    preparation_filepath = session.get('preparation_filepath')

    # 检查文件路径是否存在
    if not preparation_filepath:
        return jsonify({'error': 'Data file not found in session. Please upload and prepare the data file.'}), 400

    try:
        # 读取 CSV 文件
        df = pd.read_csv(preparation_filepath)
    except Exception as e:
        return jsonify({'error': f'Error loading data file: {str(e)}'}), 500

    # 打印调试信息，查看请求的列名和 CSV 文件中的列名
    print(f"Requested column: {column}, available columns: {df.columns.tolist()}")

    # 检查请求的列是否在数据中
    if column not in df.columns:
        return jsonify({'error': f'Column "{column}" not found in data. Available columns: {df.columns.tolist()}'}), 400

    # 获取列类型，并根据类型生成数据
    analysis_type = session.get(f'analysis_type_{column}', '维度')
    if analysis_type == '度量':
        data = {
            'labels': df[column].sort_values().tolist(),
            'values': df[column].sort_values().tolist()
        }
    else:
        data = {
            'labels': df[column].value_counts().sort_index().index.tolist(),
            'values': df[column].value_counts().sort_index().tolist()
        }

    # 返回生成的图表数据
    return jsonify(data)

@app.route('/set_session', methods=['POST'])
def set_session():
    data = request.get_json()
    filename = data.get('filename')
    if filename:
        session['uploaded_filename'] = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        session['preparation_filepath'] = session['uploaded_filename']  # 确保文件路径被更新
        print(f"Session set for filename: {filename}")  # 打印调试信息
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': 'No filename provided'})

@app.route('/api/scatter-plot', methods=['GET'])
def scatter_plot():
    x_column = request.args.get('x_column')
    y_column = request.args.get('y_column')
    preparation_filepath = session.get('preparation_filepath')
    df = pd.read_csv(preparation_filepath)

    csv_path = os.path.join('static', 'scatter_data.csv')
    df[[x_column, y_column]].to_csv(csv_path, index=False)

    return jsonify({'csv_url': url_for('static', filename='scatter_data.csv')})

def perform_feature_analysis(feature, df, x_column, y_column):
    try:
        # 针对fly ash列的特殊处理
        if x_column == 'fly_ash' or y_column == 'fly_ash':
            zero_mask = (df[x_column] == 0) if x_column == 'fly_ash' else (df[y_column] == 0)
            non_zero_df = df[~zero_mask]  # 非零值部分的数据
            zero_df = df[zero_mask]  # 零值部分的数据

            if len(non_zero_df) == 0:
                return {'error': f'All values in {x_column} or {y_column} are zero, analysis might be invalid.'}, 400

            # 对非零部分的数据进行正常的分析处理
            analysis_results_non_zero = normal_analysis_process(feature, non_zero_df, x_column, y_column)

            # 对零值部分进行处理，可以返回一些描述性统计或者跳过处理
            analysis_results_zero = {'description': 'The analysis skipped zero values as they are treated as a special case.'}

            # 合并结果
            analysis_results = {
                'non_zero_analysis': analysis_results_non_zero,
                'zero_value_handling': analysis_results_zero
            }
            return analysis_results
        
        # 其他分析代码（对于非fly_ash的处理）
        analysis_results = normal_analysis_process(feature, df, x_column, y_column)
        return analysis_results

    except KeyError as e:
        logging.error(f"KeyError in perform_feature_analysis: {str(e)}")
        return {'error': f'KeyError: {str(e)}'}, 500
    except Exception as e:
        logging.error(f"Error in perform_feature_analysis: {str(e)}")
        return {'error': f'Error processing feature {feature}: {str(e)}'}, 500

def normal_analysis_process(feature, df, x_column, y_column):
    if feature == 'CrossMeasureCorrelation':
        score, para = cal_CrossMeasureCorrelation([x_column], [x_column, y_column], df, 'sum', 'cn')
        if score > 0:
            return {
                'description': 'Cross Measure Correlation Analysis',
                'score': round(score, 4),
                'details': para.get('explain', ''),
                'para': {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in para.items()}
            }
        else:
            return {'description': 'No significant result found for Cross Measure Correlation.'}

    elif feature == 'residuals':
        alpha, beta = 1, 1
        residuals = func_residuals(df[x_column], alpha, beta)
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        max_residual = np.max(residuals)
        min_residual = np.min(residuals)
        
        score = (1 - mean_residual) * 100

        return {
            'description': f'Residuals Analysis between {x_column} and {y_column}',
            'score': round(score, 4),
            'details': f'Mean Residual: {mean_residual:.4f}, Standard Deviation: {std_residual:.4f}, Max Residual: {max_residual:.4f}, Min Residual: {min_residual:.4f}'
        }
    
    return {'description': f"No significant result found for feature {feature}."}

@app.route('/data-preparation.html', methods=['GET', 'POST'])
def data_preparation():
    message = ""
    columns = []
    column_types = {}
    analysis_types = {}
    unique_values = {}

    filepath = session.get('uploaded_filename')
    if not filepath:
        message = "文件未上传，请上传文件。"
        return render_template('data-preparation.html', message=message, columns=columns, column_types=column_types, analysis_types=analysis_types)

    df = pd.read_csv(filepath, encoding='utf-8')
    columns = df.columns.tolist()
    
    column_types = detect_column_types(df)
    analysis_types = detect_analysis_types(df)
    # 获取每个“维度”列的唯一值
    for column in columns:
        if analysis_types.get(column) == '维度':
            unique_values[column] = df[column].unique().tolist()
            if f'selected_values_{column}' not in session:
                session[f'selected_values_{column}'] = unique_values[column]
    
    if 'preparation_filepath' not in session or not session['preparation_filepath']:
        session['preparation_filepath'] = filepath

    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'prepare_data':
            fill_na = request.form.get('fill_na')
            drop_na = request.form.get('drop_na') == 'on'

            if drop_na:
                df.dropna(inplace=True)

            if fill_na != 'None':
                fill_methods = {
                    'mean': df.mean(),
                    'median': df.median(),
                    'mode': df.mode().iloc[0],
                }
                fill_value = fill_methods[fill_na]
                df.fillna(fill_value, inplace=True)

            prepared_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'prepared_{os.path.basename(filepath)}')
            df.to_csv(prepared_filepath, index=False)
            session['preparation_filepath'] = prepared_filepath
            message = "数据准备成功！"

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'global_filter_data':
            global_quantile_low = float(request.form.get('global_quantile_low', 0))
            global_quantile_high = float(request.form.get('global_quantile_high', 1))

            # 更新会话中的全局分位数
            session['global_quantile_low'] = global_quantile_low
            session['global_quantile_high'] = global_quantile_high
            
            for column in df.columns:
                # 检查该列是否为“度量”类型
                if analysis_types.get(column) == '度量':  # 仅对“度量”列执行分位数筛选
                    quantile_low_value = df[column].quantile(global_quantile_low)
                    quantile_high_value = df[column].quantile(global_quantile_high)
                    df = df[(df[column] >= quantile_low_value) & (df[column] <= quantile_high_value)]
                    
                    # 更新每列的分位数信息到 session
                    session[f'quantile_low_{column}'] = global_quantile_low
                    session[f'quantile_high_{column}'] = global_quantile_high
                elif analysis_types.get(column) == '维度':
                    session[f'selected_values_{column}'] = unique_values[column]

            # 保存过滤后的数据
            filtered_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'filtered_{os.path.basename(filepath)}')
            df.to_csv(filtered_filepath, index=False)
            session['preparation_filepath'] = filtered_filepath
            message = "全局数据筛选成功！"

        elif action == 'filter_data_column':
            # 按列筛选逻辑
            filter_column = request.form.get('filter_column')
            analysis_type = analysis_types.get(filter_column)
            if analysis_type == "度量":
                quantile_low = float(request.form.get(f'quantile_low_{filter_column}', 0))
                quantile_high = float(request.form.get(f'quantile_high_{filter_column}', 1))
            
                # 更新单列的分位数到 session
                session[f'quantile_low_{filter_column}'] = quantile_low
                session[f'quantile_high_{filter_column}'] = quantile_high

                for column in df.columns:
                    if analysis_types.get(column) == '度量':  # 仅对“度量”列执行分位数筛选
                        quantile_low = session.get(f'quantile_low_{column}', 0.0)
                        quantile_high = session.get(f'quantile_high_{column}', 1.0)
                        if column and 0 <= quantile_low <= 1 and 0 <= quantile_high <= 1 and quantile_low < quantile_high:
                    
                            low_value = df[column].quantile(quantile_low)
                            high_value = df[column].quantile(quantile_high)
                            df = df[(df[column] >= low_value) & (df[column] <= high_value)]
                    elif analysis_types.get(column) == '维度':
                        selected_values = session.get(f'selected_values_{column}', [])
                        df = df[df[column].isin(selected_values)]
            elif analysis_type == '维度':  # 对维度类型进行复选框筛选
                selected_values = request.form.getlist(f'selected_values_{filter_column}')  # 获取用户选择的值

                session[f'selected_values_{filter_column}'] = selected_values
                for column in df.columns:
                    if analysis_types.get(column) == '度量':  # 仅对“度量”列执行分位数筛选
                        quantile_low = session.get(f'quantile_low_{column}', 0.0)
                        quantile_high = session.get(f'quantile_high_{column}', 1.0)
                        if column and 0 <= quantile_low <= 1 and 0 <= quantile_high <= 1 and quantile_low < quantile_high:
                    
                            low_value = df[column].quantile(quantile_low)
                            high_value = df[column].quantile(quantile_high)
                            df = df[(df[column] >= low_value) & (df[column] <= high_value)]
                    elif analysis_types.get(column) == '维度':
                        selected_values = session.get(f'selected_values_{column}', [])
                        df = df[df[column].isin(selected_values)]
            
            filtered_filepath = os.path.join(app.config['UPLOAD_FOLDER'],
                                            f'filtered_{filter_column}_{os.path.basename(filepath)}')
            df.to_csv(filtered_filepath, index=False)
            session['preparation_filepath'] = filtered_filepath
            message = f"{filter_column} 列数据筛选成功！"

            
        elif action == 'use_column':
    # 更新是否使用该列的逻辑
            for column in columns:
        # 检查复选框是否被勾选
                use_in_analysis = request.form.get(f'use_in_analysis_{column}') == 'on'  # 如果复选框被勾选，则为 True
                session[f'use_{column}'] = use_in_analysis  # 存储选择状态
                print(session[f'use_{column}'])

            message = "列的使用状态更新成功！"
    
    return render_template('data-preparation.html', message=message, columns=columns, column_types=column_types, analysis_types=analysis_types,
                           quantile_low=session.get('global_quantile_low', 0.0), quantile_high=session.get('global_quantile_high', 1.0),
                           unique_values=unique_values)

@app.route('/update_use_column', methods=['POST'])
def update_use_column():
    data = request.get_json()
    column = data.get('column')
    use = data.get('use')

    if column:
        session[f'use_{column}'] = use  # 更新会话中的列使用状态
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': 'Column not specified'})
@app.route('/statistical-analysis.html')
def statistical_analysis():
    preparation_filepath = session.get('preparation_filepath')
    if not preparation_filepath:
        flash('No data prepared')
        return redirect(url_for('data_preparation'))

    # 读取列名，并将其传递到模板
    df = pd.read_csv(preparation_filepath)
    columns = [column for column in df.columns if session.get(f'use_{column}', True)]

    return render_template('statistical-analysis.html', columns=columns)

@app.route('/api/generate-chart', methods=['POST'])
def generate_chart():
    data = request.json
    preparation_filepath = session.get('preparation_filepath')

    # 打印准备文件路径和接收到的数据print("Preparation file path:", preparation_filepath)
    print("Received data:", data)

    if not preparation_filepath or not os.path.exists(preparation_filepath):
        return jsonify({'error': 'Data file not found'}), 400
    try:
        # 读取数据文件
        df = pd.read_csv(preparation_filepath)
    except Exception as e:
        # 如果读取文件出错，打印错误信息print(f"Error reading the CSV file: {e}")
        return jsonify({'error': f'Error reading the data file: {e}'}), 500

    x_axis = data.get('x')
    y_axis = data.get('y')
    color = data.get('color')
    size = data.get('size')

    # 检查是否选择了X轴和Y轴
    if not x_axis or not y_axis:
     return jsonify({'error': 'X and Y axes are required'}), 400
# 检查X轴和Y轴是否存在于数据集中
    if x_axis not in df.columns or y_axis not in df.columns:
        print(f"Error: Columns {x_axis} or {y_axis} not found in data. Available columns: {df.columns.tolist()}")
        return jsonify({'error': f'Columns {x_axis} or {y_axis} not found in data.'}), 400# 选择需要绘图的列
    selected_columns = [col for col in [x_axis, y_axis, color, size] if col and col in df.columns]
    plot_data = df[selected_columns].to_dict(orient='records')

    # 打印将要发送给前端的绘图数据print("Plot Data:", plot_data)
    print(f"Selected columns: {selected_columns}")

    return jsonify({
        'plot_data': plot_data,
        'x_axis': x_axis,
        'y_axis': y_axis,
        'color': color,
        'size': size
    })

@app.route('/api/get-data', methods=['GET'])
def get_data():
    try:
        preparation_filepath = session.get('preparation_filepath')
        print(f"Attempting to read file from: {preparation_filepath}")
        data = pd.read_csv(preparation_filepath)
        print(data.head())
        data_json = data.to_dict(orient='records')
        return jsonify(data_json)
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
        return jsonify({'error': '数据文件未找到，请检查路径'}), 404
    except pd.errors.EmptyDataError as e:
        print(f"EmptyDataError: {e}")
        return jsonify({'error': '文件内容为空或格式不正确'}), 400
    except Exception as e:
        print(f"Unhandled Exception: {e}")
        return jsonify({'error': str(e)}), 500
    
    # 计算统计信息的函数
def get_statistics_from_csv(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 初始化存储统计信息的列表
    statistics = []
    
    # 遍历每一列（变量）
    for column in df.columns:
        column_data = df[column]
        
        # 判断是否为类别类型（非数字类型）
        if column_data.dtype == 'object' or column_data.nunique() < 10:  # 假设少于10个唯一值的视为类别类型
            # 类别类型的统计值默认为0
            stats = {
                'name': column,
                'mean': 0,
                'stdDev': 0,
                'median': 0,
                'type': 'categorical'  # 标明为类别类型
            }
        else:
            # 数值类型的统计信息
            stats = {
                'name': column,
                'mean': column_data.mean(),
                'stdDev': column_data.std(),
                'median': column_data.median(),
                'type': 'numerical'  # 标明为数值类型
            }
        
        statistics.append(stats)
    
    return statistics

# 定义API接口，返回统计信息
@app.route('/api/get-statistics', methods=['GET'])
def get_statistics():
    try:
        # 获取CSV的统计信息
        preparation_filepath = session.get('preparation_filepath')
        stats = get_statistics_from_csv(preparation_filepath)
        return jsonify(stats)
    except Exception as e:
        # 错误处理
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/get-var', methods=['GET'])
def get_var():
    try:
        # 获取CSV的统计信息
        preparation_filepath = session.get('preparation_filepath')
        stats = get_statistics_from_csv(preparation_filepath)
        selected_vars = selected_var.split(',')
    
        # 过滤statistics列表，只保留name在selected_vars中的项
        filtered_stats = [stat for stat in stats if stat['name'] in selected_vars]
    
        return jsonify(filtered_stats)
    except Exception as e:
        # 错误处理
        return jsonify({"error": str(e)}), 500



# 检查分析状态的路由
@app.route('/check-analysis-status', methods=['GET'])
def check_analysis_status():
    flag_file_path = "in_progress.flag"
    if os.path.exists(flag_file_path):
        return jsonify({"completed": False})
    else:
        return jsonify({"completed": True})
    

@app.route('/causal-analysis.html', methods=['GET', 'POST'])
def causal_analysis_view():
    csv_data = []
    output_file_path = os.path.join('static', 'result.csv')

    if request.method == 'POST':
        # 从会话中获取数据文件路径
        data_file_path = session.get('preparation_filepath')
        print(f"Using data file path from session: {data_file_path}")
        
        if not data_file_path:
            error_message = "数据文件未找到，请先上传并准备数据文件。"
            return render_template('causal-analysis.html', error_message=error_message), 400

        # 获取算法类型
        algorithm = request.form.get('algorithm')

        background_edge_json = request.form.get('background_edge')
        background_edge = json.loads(background_edge_json) if background_edge_json else []
        print(f"接收到的 background_edge: {background_edge}")

        global selected_var
        selected_var= request.form.get('sel_var')
        selected_var = selected_var if selected_var else ""
        print("+++++++++++++++++++++++")
        print(selected_var)
        print(type(selected_var))

        # 创建标志文件，表示分析开始
        flag_file_path = "in_progress.flag"
        with open(flag_file_path, 'w') as f:
            f.write("Analysis in progress")

        # 启动分析任务
        try:
            if algorithm == 'pc':
                print("now is pc")
                script_path = 'PC算法/pc_easy.py'
                result = subprocess.run(
                    ['python', script_path, '--data_file', data_file_path, '--output_file', output_file_path, '--background_edge', background_edge_json,'--variable_names',selected_var],
                    capture_output=True, text=True
                )
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)

            elif algorithm == 'gies':
                print("now is gies")
                script_path = 'GIES算法/gies_easy.py'
                result = subprocess.run(
                    ['python', script_path, '--data_file', data_file_path, '--output_file', output_file_path, '--background_edge', background_edge_json,'--variable_names',selected_var],
                    capture_output=True, text=True
                )

            # 打印调试输出
            print(result.stdout)
            print(selected_var)

            # 检查返回状态
            if result.returncode != 0:
                error_message = f"因果分析失败: {result.stderr}"
                return render_template('causal-analysis.html', error_message=error_message), 500

            # 检查结果文件是否生成
            if os.path.exists(output_file_path):
                df = pd.read_csv(output_file_path)
                csv_data = df.to_dict(orient='records')
                return jsonify(csv_data=csv_data)  # 将数据返回给前端

            else:
                error_message = "生成的结果文件不存在。"
                return render_template('causal-analysis.html', error_message=error_message), 500

        finally:
            # 删除标志文件，表示分析完成
            if os.path.exists(flag_file_path):
                os.remove(flag_file_path)
            print("Analysis completed, flag file removed.")

        print(output_file_path)
        print(csv_data)
        return render_template('causal-analysis.html', csv_data=csv_data)

    # GET 请求时返回空的结果
    return render_template('causal-analysis.html', csv_data=csv_data)


@app.route('/get-csv-data', methods=['GET'])
def get_csv_data():
    output_file_path = os.path.join('static', 'result.csv')

    if not os.path.exists(output_file_path):
        return jsonify({"error": "结果文件未找到"}), 404

    # 读取 CSV 文件
    df = pd.read_csv(output_file_path)
    csv_data = df.to_dict(orient='records')  # 用于表格展示

    # 提取 Node1 和 Node2 列
    if 'Node1' in df.columns and 'Node2' in df.columns:
        nodes = list(set(df['Node1']).union(set(df['Node2'])))
        edges = [{'source': row['Node1'], 'target': row['Node2']} for _, row in df.iterrows()]
    else:
        nodes, edges = [], []

    return jsonify({"csv_data": csv_data, "nodes": nodes, "edges": edges})


@app.route('/get_csv_data_new', methods=['GET'])
def get_csv_data_new():
    output_file_path = os.path.join('static', 'result_new.csv')

    if not os.path.exists(output_file_path):
        return jsonify({"error": "结果文件未找到"}), 404

    # 读取 CSV 文件
    df = pd.read_csv(output_file_path)
    csv_data = df.to_dict(orient='records')  # 用于表格展示

    return jsonify(csv_data)

@app.route('/big-model-analysis.html')
def big_model_analysis():
    return render_template('big-model-analysis.html')

@app.route('/favorites.html')
def favorites():
    return render_template('favorites.html')

@app.route('/uploads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download/<filename>')
def download(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    else:
        return "文件未找到", 404
    
@app.route('/api/chat', methods=['POST'])
async def chat_handler():
    """处理大模型对话请求"""
    try:
        print("++++++++++++++++++++++=")
        print(selected_var)
        variable_names = selected_var.split(',')
        print(variable_names)

        # 读取指定列的数据
        df = pd.read_csv(session.get('preparation_filepath'), usecols=variable_names)
    
        # 初始化存储统计信息的列表
        statistics = []
        
        # 遍历每一列（变量）
        for column in df.columns:
            statistics.append(column)
        causal_graph = client.generate_causal_graph(statistics)
        response_data = {
            "nodes": list(causal_graph.nodes()),
            "edges": [
                {"from": src, "to": dst} 
                for src, dst in causal_graph.edges()
            ]
        }
        
        return jsonify(response_data), 200
        
        
    except Exception as e:
        logging.error(f"API Error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@app.route('/analyze_two_factor', methods=['POST'])
def analyze_two_factor():
    # 获取请求中的JSON数据
    data = request.get_json()

    # 从JSON数据中提取变量
    cause_var = data.get('cause_var')
    effect_var = data.get('effect_var')

    # 打印或处理这些变量
    print(f"原因变量: {cause_var}, 结果变量: {effect_var}")

    return jsonify({"status": "success", "message": "Variables received"}), 200


@app.route('/api/backdoor_adjustment', methods=['POST'])
def backdoor_adjustment():
    data = request.get_json()
    return client.backdoorAdjustment(data)
    


@app.route('/api/calculate-effect', methods=['POST'])
def calculate_effect():
    try:
        # 读取CSV文件并创建图
        G = nx.DiGraph()
        df = pd.read_csv('static/result_new.csv')
        edges = []
        for _, row in df.iterrows():
            node1 = row['Node1']
            node2 = row['Node2']
            edges.append((node1, node2))
        G.add_edges_from(edges)

        # 验证图是否是DAG
        if nx.is_directed_acyclic_graph(G):
            print("图验证通过：是有向无环图（DAG）")
        else:
            return jsonify({"error": "图中存在环，无法计算因果效应！"}), 400

        # 获取文件路径并处理数据
        data_file_path = session.get('preparation_filepath')  # 注意这里你可以考虑使用请求体传递文件路径
        t = EffectiveInformation(data_file_path, G)

        data = request.get_json()

        #从JSON数据中提取变量
        cause_var = data.get('cause_var')
        effect_var = data.get('effect_var')

        # 设置处理变量和目标变量
        success = t.set_var(
            treatment_name=cause_var, 
            target_name=effect_var
        )

        if success:
            # 计算因果效应指标
            effects = t.measure_causal_effect()

            # 构造返回的数据字典
            result = {
                "kl_divergence": effects[0],
                "js_divergence": effects[1],
                "total_variation": effects[2],
                "wasserstein_distance": effects[3],
                "hellinger_distance": effects[4]
            }
            print(result)

            return jsonify(result)  # 返回计算结果
        else:
            return jsonify({"error": "因果路径不存在，请检查图结构！"}), 400

    except Exception as e:
        # 捕获异常并返回错误信息
        print("发生错误:", str(e))
        return jsonify({"error": "发生错误，请检查服务器日志"}), 500



if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
