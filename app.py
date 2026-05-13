import os
import pandas as pd
import numpy as np
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
import warnings
from sqlalchemy.orm import joinedload
warnings.filterwarnings('ignore')
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for, abort
from functools import wraps
import webbrowser
import threading
import time

# ========== 导入自定义转换模块 ==========
from convert_to_csv import process_single_txt_to_csv, parse_txt_data

# ========== 解决中文乱码 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== Flask 应用配置 ==========
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CSV_FOLDER'] = '温湿度数据csv'
app.config['STATIC_FOLDER'] = 'static'
app.config['OUTPUT_FOLDER'] = '预测结果'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CSV_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# ========== 数据库初始化 ==========
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ========== 模型定义 ==========
class SystemConfig(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(100), unique=True, nullable=False)
    value = db.Column(db.String(500))
    description = db.Column(db.String(200))
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class LoginLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    username = db.Column(db.String(80))
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(300))
    success = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    status = db.Column(db.String(20), default='active')
    role = db.Column(db.String(20), default='user')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    receiver_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    is_read = db.Column(db.Boolean, default=False)

    sender = db.relationship('User', foreign_keys=[sender_id], backref='sent_messages')
    receiver = db.relationship('User', foreign_keys=[receiver_id], backref='received_messages')

class UploadHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    original_filename = db.Column(db.String(200))
    target = db.Column(db.String(50))
    model_used = db.Column(db.String(50))
    pred_days = db.Column(db.Integer)
    data_span_days = db.Column(db.Integer)
    csv_path = db.Column(db.String(300))
    img_path = db.Column(db.String(300))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref='histories')

class AdminMaterial(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200))
    content = db.Column(db.Text)
    file_path = db.Column(db.String(300))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ========== 权限装饰器 ==========
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        if current_user.role != 'admin':
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

def operator_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        if current_user.role not in ['admin', 'operator']:
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

def get_config(key, default=None):
    config = SystemConfig.query.filter_by(key=key).first()
    return config.value if config else default

def set_config(key, value, description=''):
    config = SystemConfig.query.filter_by(key=key).first()
    if config:
        config.value = value
        config.updated_at = datetime.utcnow()
    else:
        config = SystemConfig(key=key, value=value, description=description)
        db.session.add(config)
    db.session.commit()

# 创建数据库表
with app.app_context():
    db.create_all()
    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin', role='admin')
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()

# ========== 预测核心函数 ==========
def get_data_span_from_df(df, date_col='日期'):
    if date_col not in df.columns:
        return 0
    dates = pd.to_datetime(df[date_col])
    return (dates.max() - dates.min()).days

def aggregate_sensors_from_dir(input_dir, target_col, freq='h'):
    all_dfs = []
    for filename in os.listdir(input_dir):
        if not filename.endswith('.csv'):
            continue
        filepath = os.path.join(input_dir, filename)
        df = pd.read_csv(filepath, parse_dates=['日期'], encoding='utf-8-sig')
        if target_col not in df.columns:
            continue
        sub = df[['日期', target_col]].set_index('日期')
        sub = sub.resample(freq).mean()
        all_dfs.append(sub)
    if not all_dfs:
        raise ValueError(f"未找到目标列 '{target_col}' 的数据")
    combined = pd.concat(all_dfs, axis=1)
    agg_series = combined.mean(axis=1, skipna=True)
    agg_series = agg_series.ffill()
    agg_df = agg_series.reset_index()
    agg_df.columns = ['ds', 'y']
    return agg_df

def predict_prophet(df, target_name, periods=30*24, save_img_path=None):
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='h')
    forecast = model.predict(future)
    fig = model.plot(forecast)
    plt.title(f'{target_name} 未来 {periods//24} 天预测 (Prophet)')
    plt.xlabel('时间')
    plt.ylabel(target_name)
    if save_img_path:
        plt.savefig(save_img_path, dpi=150, bbox_inches='tight')
    plt.close()
    pred = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    pred.columns = ['时间', '预测值', '下限', '上限']
    return pred

def predict_linear(df, target_name, periods=7*24, window=24, save_img_path=None):
    series = df['y'].values
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    X = np.array(X)
    y = np.array(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_scaled, y)
    last_window = series[-window:].copy()
    predictions = []
    for _ in range(periods):
        last_scaled = scaler.transform(last_window.reshape(1, -1))
        pred = model.predict(last_scaled)[0]
        predictions.append(pred)
        last_window = np.append(last_window[1:], pred)
    last_date = df['ds'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(1, unit='h'), periods=periods, freq='h')
    if target_name == '温度(℃)':
        predictions = np.clip(predictions, 0, 40)
        lower = np.clip(predictions - 2, 0, 40)
        upper = np.clip(predictions + 2, 0, 40)
    else:
        predictions = np.clip(predictions, 0, 100)
        lower = np.clip(predictions - 5, 0, 100)
        upper = np.clip(predictions + 5, 0, 100)
    pred_df = pd.DataFrame({
        '时间': future_dates,
        '预测值': predictions,
        '下限': lower,
        '上限': upper
    })
    plt.figure(figsize=(12, 6))
    plt.plot(df['ds'], df['y'], label='历史数据')
    plt.plot(pred_df['时间'], pred_df['预测值'], label='预测值', color='red')
    plt.fill_between(pred_df['时间'], pred_df['下限'], pred_df['上限'], alpha=0.2, color='red')
    plt.title(f'{target_name} 未来 {periods//24} 天预测 (线性回归)')
    plt.xlabel('时间')
    plt.ylabel(target_name)
    plt.legend()
    plt.grid(True)
    if save_img_path:
        plt.savefig(save_img_path, dpi=150, bbox_inches='tight')
    plt.close()
    return pred_df

def perform_prediction(data_df, target, model_choice, span_days, prefix=None):
    df = data_df[['日期', target]].rename(columns={'日期': 'ds', target: 'y'})
    df = df.set_index('ds').resample('h').mean().reset_index()
    df['y'] = df['y'].ffill()

    prophet_days = int(get_config('prediction.prophet_days', '30'))
    linear_days = int(get_config('prediction.linear_days', '7'))
    auto_threshold = int(get_config('prediction.auto_threshold_days', '365'))

    if model_choice == 'auto':
        use_prophet = span_days >= auto_threshold
    else:
        use_prophet = (model_choice == 'prophet')

    if use_prophet:
        periods = prophet_days * 24
        model_name = "Prophet (年周期)"
        pred_df = predict_prophet(df, target, periods=periods)
    else:
        periods = linear_days * 24
        model_name = "线性回归 (滑动窗口)"
        pred_df = predict_linear(df, target, periods=periods)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if prefix:
        img_filename = f"{prefix}_{target}_{periods//24}天.png"
        csv_filename = f"{prefix}_{target}_{periods//24}天.csv"
    else:
        img_filename = f"pred_{timestamp}_{target}.png"
        csv_filename = f"pred_{timestamp}_{target}.csv"
    img_path = os.path.join(app.config['STATIC_FOLDER'], img_filename)
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)

    if use_prophet:
        predict_prophet(df, target, periods=periods, save_img_path=img_path)
    else:
        predict_linear(df, target, periods=periods, save_img_path=img_path)
    pred_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    return pred_df, model_name, periods//24, img_path, csv_path

# ========== 基础路由 ==========
@app.route('/')
def index():
    return redirect(url_for('home'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        ip = request.remote_addr
        ua = request.user_agent.string

        if user and user.check_password(password):
            if user.status == 'disabled':
                flash('账号已被禁用，请联系管理员')
                log = LoginLog(user_id=user.id, username=username, ip_address=ip, user_agent=ua, success=False)
                db.session.add(log)
                db.session.commit()
                return render_template('login.html')
            login_user(user)
            log = LoginLog(user_id=user.id, username=username, ip_address=ip, user_agent=ua, success=True)
            db.session.add(log)
            db.session.commit()
            return redirect(url_for('home'))
        else:
            log = LoginLog(username=username, ip_address=ip, user_agent=ua, success=False)
            db.session.add(log)
            db.session.commit()
            flash('用户名或密码错误')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('用户名已存在')
        else:
            user = User(username=username)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash('注册成功，请登录')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/home')
@login_required
def home():
    materials = AdminMaterial.query.order_by(AdminMaterial.created_at.desc()).all()
    return render_template('home.html', user=current_user, materials=materials)

@app.route('/admin/upload_material', methods=['POST'])
@login_required
@operator_required
def upload_material():
    title = request.form['title']
    content = request.form['content']
    file = request.files.get('file')
    file_path = None
    if file and file.filename:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
    material = AdminMaterial(title=title, content=content, file_path=file_path)
    db.session.add(material)
    db.session.commit()
    flash('资料已上传')
    return redirect(url_for('home'))

# ========== 预测中心 ==========
@app.route('/predict_center', methods=['GET', 'POST'])
@login_required
def predict_center():
    if request.method == 'POST':
        target = request.form.get('target', '温度(℃)')
        model_choice = request.form.get('model_choice', 'auto')
        uploaded_files = request.files.getlist('files')
        if not uploaded_files or all(f.filename == '' for f in uploaded_files):
            flash('请至少上传一个 .txt 文件')
            return redirect(url_for('predict_center'))

        results = []

        for uploaded_file in uploaded_files:
            if not uploaded_file.filename.endswith('.txt'):
                flash(f'文件 {uploaded_file.filename} 不是 .txt 格式，已跳过')
                continue

            original_filename = uploaded_file.filename
            import uuid
            temp_filename = f"{uuid.uuid4().hex}.txt"
            temp_txt_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
            uploaded_file.save(temp_txt_path)

            try:
                base_name = os.path.splitext(original_filename)[0]
                csv_path = process_single_txt_to_csv(
                    temp_txt_path,
                    output_dir=app.config['CSV_FOLDER'],
                    output_filename=base_name
                )
                df = pd.read_csv(csv_path, parse_dates=['日期'])
                span_days = get_data_span_from_df(df)

                pred_df, model_name, pred_days, img_path, out_csv_path = perform_prediction(
                    df, target, model_choice, span_days, prefix=base_name
                )

                history = UploadHistory(
                    user_id=current_user.id,
                    original_filename=original_filename,
                    target=target,
                    model_used=model_name,
                    pred_days=pred_days,
                    data_span_days=span_days,
                    csv_path=out_csv_path,
                    img_path=img_path
                )
                db.session.add(history)
                db.session.commit()

                results.append({
                    'type': 'file',
                    'filename': original_filename,
                    'success': True,
                    'img_url': f'/static/{os.path.basename(img_path)}',
                    'csv_url': f'/download_history/{history.id}',
                    'model_name': model_name,
                    'pred_days': pred_days,
                    'span_days': span_days
                })
            except Exception as e:
                results.append({
                    'type': 'file',
                    'filename': original_filename,
                    'success': False,
                    'error': str(e)
                })
            finally:
                if os.path.exists(temp_txt_path):
                    os.remove(temp_txt_path)

        # 整体大棚预测
        try:
            agg_df = aggregate_sensors_from_dir(app.config['CSV_FOLDER'], target, freq='h')
            span_days = (agg_df['ds'].max() - agg_df['ds'].min()).days
            df_for_pred = agg_df.rename(columns={'ds': '日期', 'y': target})
            pred_df, model_name, pred_days, img_path, out_csv_path = perform_prediction(
                df_for_pred, target, model_choice, span_days, prefix="整体大棚"
            )
            temp_csv_filename = os.path.basename(out_csv_path)
            results.append({
                'type': 'greenhouse',
                'filename': '整体大棚（所有传感器）',
                'success': True,
                'img_url': f'/static/{os.path.basename(img_path)}',
                'csv_url': f'/download_temp/{temp_csv_filename}',
                'model_name': model_name,
                'pred_days': pred_days,
                'span_days': span_days
            })
        except Exception as e:
            results.append({
                'type': 'greenhouse',
                'filename': '整体大棚',
                'success': False,
                'error': str(e)
            })

        return render_template('batch_result.html', results=results, target=target)

    system_span = 0
    if os.path.exists(app.config['CSV_FOLDER']):
        all_dates = []
        for f in os.listdir(app.config['CSV_FOLDER']):
            if f.endswith('.csv'):
                df_tmp = pd.read_csv(os.path.join(app.config['CSV_FOLDER'], f), parse_dates=['日期'])
                all_dates.extend(df_tmp['日期'].tolist())
        if all_dates:
            system_span = (max(all_dates) - min(all_dates)).days

    auto_threshold = int(get_config('prediction.auto_threshold_days', '365'))
    return render_template('predict_center.html',
                           system_span=system_span,
                           auto_threshold=auto_threshold)

@app.route('/download_temp/<filename>')
@login_required
def download_temp(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        flash('文件不存在')
        return redirect(url_for('profile'))
    return send_file(filepath, as_attachment=True)

# ========== 个人中心 ==========
@app.route('/profile')
@login_required
def profile():
    history = UploadHistory.query.filter_by(user_id=current_user.id).order_by(UploadHistory.created_at.desc()).all()
    return render_template('profile.html', user=current_user, history=history)

@app.route('/change_password', methods=['POST'])
@login_required
def change_password():
    old = request.form['old_password']
    new = request.form['new_password']
    if current_user.check_password(old):
        current_user.set_password(new)
        db.session.commit()
        flash('密码已修改')
    else:
        flash('原密码错误')
    return redirect(url_for('profile'))

@app.route('/download_history/<int:history_id>')
@login_required
def download_history(history_id):
    history = UploadHistory.query.get_or_404(history_id)
    if history.user_id != current_user.id and current_user.role not in ['admin', 'operator']:
        flash('无权下载')
        return redirect(url_for('profile'))
    return send_file(history.csv_path, as_attachment=True)

@app.route('/delete_history/<int:history_id>', methods=['POST'])
@login_required
def delete_history(history_id):
    history = UploadHistory.query.get_or_404(history_id)
    if history.user_id != current_user.id and current_user.role not in ['admin', 'operator']:
        flash('无权删除')
        return redirect(url_for('profile'))
    try:
        if os.path.exists(history.csv_path):
            os.remove(history.csv_path)
        if os.path.exists(history.img_path):
            os.remove(history.img_path)
    except:
        pass
    db.session.delete(history)
    db.session.commit()
    flash('已删除')
    return redirect(url_for('profile'))

@app.route('/clear_history', methods=['POST'])
@login_required
def clear_history():
    histories = UploadHistory.query.filter_by(user_id=current_user.id).all()
    for history in histories:
        try:
            if os.path.exists(history.csv_path):
                os.remove(history.csv_path)
            if os.path.exists(history.img_path):
                os.remove(history.img_path)
        except:
            pass
        db.session.delete(history)
    db.session.commit()
    flash('已清空所有历史记录')
    return redirect(url_for('profile'))

# ========== 消息系统 ==========
@app.route('/messages')
@login_required
def messages():
    received = Message.query.filter_by(receiver_id=current_user.id)\
        .options(joinedload(Message.sender)).order_by(Message.timestamp.desc()).all()
    sent = Message.query.filter_by(sender_id=current_user.id)\
        .options(joinedload(Message.receiver)).order_by(Message.timestamp.desc()).all()
    users = User.query.filter(User.id != current_user.id).all()
    return render_template('messages.html', received=received, sent=sent, users=users)

@app.route('/send_message', methods=['POST'])
@login_required
def send_message():
    receiver_name = request.form['receiver']
    content = request.form['content']
    if not content.strip():
        flash('消息内容不能为空')
        return redirect(url_for('messages'))
    receiver = User.query.filter_by(username=receiver_name).first()
    if not receiver:
        flash('用户不存在')
        return redirect(url_for('messages'))
    if receiver.id == current_user.id:
        flash('不能给自己发送消息')
        return redirect(url_for('messages'))
    msg = Message(sender_id=current_user.id, receiver_id=receiver.id, content=content.strip())
    db.session.add(msg)
    db.session.commit()
    flash('消息已发送')
    return redirect(url_for('messages'))

# ========== 管理员用户管理（仅 admin） ==========
@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    users = User.query.order_by(User.id).all()
    return render_template('admin_users.html', users=users)

@app.route('/admin/user/toggle/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def admin_user_toggle(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash('不能修改自己的状态')
        return redirect(url_for('admin_users'))
    user.status = 'disabled' if user.status == 'active' else 'active'
    db.session.commit()
    flash(f'用户 {user.username} 状态已更新')
    return redirect(url_for('admin_users'))

@app.route('/admin/user/role/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def admin_user_role(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash('不能修改自己的角色')
        return redirect(url_for('admin_users'))
    new_role = request.form.get('role')
    if new_role in ['admin', 'operator', 'viewer', 'user']:
        user.role = new_role
        db.session.commit()
        flash(f'用户 {user.username} 角色已更新为 {new_role}')
    return redirect(url_for('admin_users'))

@app.route('/admin/users/import', methods=['POST'])
@login_required
@admin_required
def admin_users_import():
    file = request.files.get('csv_file')
    if not file or not file.filename.endswith('.csv'):
        flash('请上传CSV文件')
        return redirect(url_for('admin_users'))
    try:
        df = pd.read_csv(file, encoding='utf-8-sig')
        required_cols = ['username', 'password', 'role']
        if not all(col in df.columns for col in required_cols):
            flash('CSV必须包含 username, password, role 列')
            return redirect(url_for('admin_users'))
        count = 0
        for _, row in df.iterrows():
            if User.query.filter_by(username=row['username']).first():
                continue
            user = User(username=row['username'], role=row.get('role', 'user'))
            user.set_password(str(row['password']))
            user.status = 'active'
            db.session.add(user)
            count += 1
        db.session.commit()
        flash(f'成功导入 {count} 个用户')
    except Exception as e:
        flash(f'导入失败：{str(e)}')
    return redirect(url_for('admin_users'))

@app.route('/admin/logs')
@login_required
@admin_required
def admin_logs():
    page = request.args.get('page', 1, type=int)
    pagination = LoginLog.query.order_by(LoginLog.timestamp.desc()).paginate(page=page, per_page=30)
    return render_template('admin_logs.html', pagination=pagination)

@app.route('/admin/user/delete/<int:user_id>')
@login_required
@admin_required
def admin_delete_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash('不能删除自己')
        return redirect(url_for('admin_users'))
    Message.query.filter((Message.sender_id == user.id) | (Message.receiver_id == user.id)).delete()
    UploadHistory.query.filter_by(user_id=user.id).delete()
    db.session.delete(user)
    db.session.commit()
    flash(f'用户 {user.username} 已删除')
    return redirect(url_for('admin_users'))

# ========== 数据文件管理（操作员及以上） ==========
@app.route('/admin/datafiles')
@login_required
@operator_required
def admin_datafiles():
    csv_dir = app.config['CSV_FOLDER']
    files_info = []
    for filename in os.listdir(csv_dir):
        if not filename.endswith('.csv'):
            continue
        filepath = os.path.join(csv_dir, filename)
        stat = os.stat(filepath)
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
            record_count = len(df)
            if '日期' in df.columns:
                df['日期'] = pd.to_datetime(df['日期'])
                min_date = df['日期'].min().strftime('%Y-%m-%d')
                max_date = df['日期'].max().strftime('%Y-%m-%d')
                span_days = (df['日期'].max() - df['日期'].min()).days
            else:
                min_date = max_date = span_days = 'N/A'
            null_count = df.isnull().sum().sum()
            has_issue = null_count > record_count * 0.1
        except Exception as e:
            record_count = min_date = max_date = span_days = '读取失败'
            has_issue = True
        files_info.append({
            'name': filename,
            'size': f"{stat.st_size / 1024:.1f} KB",
            'records': record_count,
            'min_date': min_date,
            'max_date': max_date,
            'span_days': span_days,
            'has_issue': has_issue,
            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
        })
    return render_template('admin_datafiles.html', files=files_info)

@app.route('/admin/datafiles/delete', methods=['POST'])
@login_required
@operator_required
def admin_datafiles_delete():
    filenames = request.form.getlist('filenames')
    csv_dir = app.config['CSV_FOLDER']
    deleted = []
    for fn in filenames:
        if '..' in fn or '/' in fn or '\\' in fn:
            continue
        filepath = os.path.join(csv_dir, fn)
        if os.path.exists(filepath) and fn.endswith('.csv'):
            os.remove(filepath)
            deleted.append(fn)
    flash(f'已删除 {len(deleted)} 个文件')
    return redirect(url_for('admin_datafiles'))

@app.route('/admin/datafiles/check/<filename>')
@login_required
@operator_required
def admin_datafiles_check(filename):
    filepath = os.path.join(app.config['CSV_FOLDER'], filename)
    try:
        col_names, kept_rows = parse_txt_data(filepath)
        return jsonify({
            'status': 'ok',
            'filename': filename,
            'columns': col_names,
            'valid_records': len(kept_rows),
            'message': f'清洗后有效记录 {len(kept_rows)} 条'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/admin/datafiles/aggregate', methods=['POST'])
@login_required
@operator_required
def admin_aggregate():
    try:
        for target in ['温度(℃)', '湿度(%)']:
            agg_df = aggregate_sensors_from_dir(app.config['CSV_FOLDER'], target, freq='h')
            cache_path = os.path.join(app.config['OUTPUT_FOLDER'], f'aggregate_{target}.csv')
            agg_df.to_csv(cache_path, index=False, encoding='utf-8-sig')
        flash('整体大棚聚合数据已更新')
    except Exception as e:
        flash(f'聚合失败：{str(e)}')
    return redirect(url_for('admin_datafiles'))

# ========== 预测历史管理（操作员及以上） ==========
@app.route('/admin/history')
@login_required
@operator_required
def admin_history():
    page = request.args.get('page', 1, type=int)
    user_filter = request.args.get('user_id', type=int)
    query = UploadHistory.query.options(joinedload(UploadHistory.user))
    if user_filter:
        query = query.filter_by(user_id=user_filter)
    pagination = query.order_by(UploadHistory.created_at.desc()).paginate(page=page, per_page=20)
    users = User.query.all()
    return render_template('admin_history.html', pagination=pagination, users=users, user_filter=user_filter)

@app.route('/admin/history/delete', methods=['POST'])
@login_required
@operator_required
def admin_history_delete():
    ids = request.form.getlist('history_ids')
    for hid in ids:
        history = UploadHistory.query.get(int(hid))
        if history:
            try:
                if os.path.exists(history.csv_path):
                    os.remove(history.csv_path)
                if os.path.exists(history.img_path):
                    os.remove(history.img_path)
            except:
                pass
            db.session.delete(history)
    db.session.commit()
    flash(f'已删除 {len(ids)} 条记录')
    return redirect(url_for('admin_history'))

@app.route('/admin/history/clear_invalid', methods=['POST'])
@login_required
@operator_required
def admin_history_clear_invalid():
    histories = UploadHistory.query.all()
    count = 0
    for h in histories:
        if not os.path.exists(h.csv_path) and not os.path.exists(h.img_path):
            db.session.delete(h)
            count += 1
    db.session.commit()
    flash(f'已清理 {count} 条无效记录')
    return redirect(url_for('admin_history'))

# ========== 系统设置（仅 admin） ==========
@app.route('/admin/settings', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_settings():
    if request.method == 'POST':
        set_config('prediction.default_model', request.form.get('default_model', 'auto'))
        set_config('prediction.prophet_days', request.form.get('prophet_days', '30'))
        set_config('prediction.linear_days', request.form.get('linear_days', '7'))
        set_config('prediction.auto_threshold_days', request.form.get('auto_threshold', '365'))
        set_config('prophet.changepoint_scale', request.form.get('changepoint_scale', '0.05'))
        flash('配置已保存')
        return redirect(url_for('admin_settings'))

    configs = {
        'default_model': get_config('prediction.default_model', 'auto'),
        'prophet_days': get_config('prediction.prophet_days', '30'),
        'linear_days': get_config('prediction.linear_days', '7'),
        'auto_threshold': get_config('prediction.auto_threshold_days', '365'),
        'changepoint_scale': get_config('prophet.changepoint_scale', '0.05'),
    }
    return render_template('admin_settings.html', configs=configs)

def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://localhost:5000/login")

if __name__ == '__main__':
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        threading.Timer(1.2, open_browser).start()
    app.run(debug=True, port=5000)