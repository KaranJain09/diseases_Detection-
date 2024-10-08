# urls.py
from django.urls import path
from myapp import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload-mri/', views.upload_mri, name='upload_mri'),
    path('upload-xray/', views.upload_xray, name='upload_xray'),
    path('upload-ct/', views.upload_ct, name='upload_ct'),
]
