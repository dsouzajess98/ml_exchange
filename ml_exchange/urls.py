from . import views
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings
from web3auth import urls as web3auth_urls

urlpatterns = [
    path('', views.login, name='login'),
    path('accounts/profile/', views.home_page, name='home_page'),
    path('new_contract/', views.new_contract, name='new_contract'),
    path('model_submission/', views.upload_model, name='upload_model'),
    path('training_data/<int:id>/', views.get_training_data, name='get_training_data'),
    path('approve_request/', views.approve_request, name='approve_request'),
    path('new_request/<int:type>/', views.new_request, name='new_request'),
    path('login/', views.login, name='login'),
    path('auto_login/', views.auto_login, name='autologin'),
    path('web3', include('web3auth.urls')),
    path("logout", views.logout_request, name="logout"),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)