# -*- coding: utf-8 -*-
"""
app.py - 后端服务
Flask应用，提供训练、预测、结果查询接口
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import torch
import numpy as np
import pandas as pd
from train import RNNModel, TimeSeriesDataset, train_all_models
import threading
import traceback

app = Flask(__name__)
CORS(app)

# 全局状态
training_status = {
    'running': False,
    'progress': 0,
    'message': '',
    'results': {}
}


@app.route('/api/train', methods=['POST'])
def start_training():
    """启动模型训练"""
    global training_status
    
    if training_status['running']:
        return jsonify({'error': '训练已在运行中'}), 400
    
    data = request.json
    dataset = data.get('dataset', 'beijing_pm25')
    epochs = data.get('epochs', 100)
    batch_size = data.get('batch_size', 64)
    hidden_size = data.get('hidden_size', 64)
    num_layers = data.get('num_layers', 2)
    
    # 在后台运行训练
    def run_training():
        global training_status
        try:
            training_status['running'] = True
            training_status['progress'] = 0
            training_status['message'] = f'开始训练 {dataset} 数据集...'
            
            dataset_path = f"./data/{dataset}.csv"
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
            
            results = train_all_models(
                dataset_path=dataset_path,
                dataset_name=dataset,
                epochs=epochs,
                batch_size=batch_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
            )
            
            training_status['results'] = {
                k: {
                    'test_mse': v['test_mse'],
                    'test_mae': v['test_mae'],
                    'test_rmse': v['test_rmse'],
                }
                for k, v in results.items()
            }
            training_status['progress'] = 100
            training_status['message'] = '训练完成！'
        except Exception as e:
            training_status['message'] = f'训练失败: {str(e)}'
            training_status['progress'] = -1
            print(traceback.format_exc())
        finally:
            training_status['running'] = False
    
    thread = threading.Thread(target=run_training)
    thread.start()
    
    return jsonify({'status': 'started', 'message': '训练已启动'})


@app.route('/api/status', methods=['GET'])
def get_status():
    """获取训练状态"""
    return jsonify(training_status)


@app.route('/api/results', methods=['GET'])
def get_results():
    """获取训练结果"""
    dataset = request.args.get('dataset', 'beijing_pm25')
    
    result_path = f"./results/summary_{dataset}.json"
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            results = json.load(f)
        return jsonify(results)
    else:
        return jsonify({'error': '未找到结果文件'}), 404


@app.route('/api/predict', methods=['POST'])
def predict():
    """使用训练好的模型进行预测"""
    data = request.json
    dataset = data.get('dataset', 'beijing_pm25')
    model_type = data.get('model', 'LSTM')
    input_data = data.get('data')  # [[1,2,3,...], [4,5,6,...], ...]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # 加载模型
        checkpoint_path = f"./checkpoints/{model_type}_{dataset}_best.pt"
        if not os.path.exists(checkpoint_path):
            return jsonify({'error': f'模型文件不存在: {checkpoint_path}'}), 404
        
        # 确定输入大小
        dataset_path = f"./data/{dataset}.csv"
        test_dataset = TimeSeriesDataset(dataset_path, split='test')
        input_size = test_dataset.data.shape[1]
        
        model = RNNModel(input_size=input_size, model_type=model_type)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        # 预处理输入
        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(0)  # 添加batch维度
        
        # 预测
        with torch.no_grad():
            output = model(input_tensor)
        
        predictions = output.cpu().numpy().tolist()
        
        return jsonify({
            'model': model_type,
            'predictions': predictions,
            'shape': list(output.shape)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    """列出可用的数据集"""
    datasets = []
    data_dir = './data'
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith('.csv'):
                name = file.replace('.csv', '')
                path = os.path.join(data_dir, file)
                size = os.path.getsize(path)
                datasets.append({'name': name, 'size': size})
    
    return jsonify({'datasets': datasets})


@app.route('/api/models', methods=['GET'])
def list_models():
    """列出已训练的模型"""
    models = {}
    checkpoint_dir = './checkpoints'
    if os.path.exists(checkpoint_dir):
        for file in os.listdir(checkpoint_dir):
            if file.endswith('.pt'):
                parts = file.replace('.pt', '').split('_')
                model_type = parts[0]
                dataset = '_'.join(parts[1:-1])
                
                if dataset not in models:
                    models[dataset] = []
                if model_type not in models[dataset]:
                    models[dataset].append(model_type)
    
    return jsonify({'models': models})


@app.route('/api/download/results/<dataset>', methods=['GET'])
def download_results(dataset):
    """下载结果文件"""
    result_path = f"./results/summary_{dataset}.json"
    if os.path.exists(result_path):
        return send_file(result_path, as_attachment=True)
    else:
        return jsonify({'error': '文件不存在'}), 404


@app.route('/')
def index():
    """返回仪表板HTML"""
    return send_file('dashboard.html')


@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({'status': 'ok', 'gpu': torch.cuda.is_available()})


if __name__ == '__main__':
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    # 开发环境下运行
    app.run(debug=True, host='0.0.0.0', port=5000)
