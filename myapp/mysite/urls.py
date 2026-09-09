from django.urls import path
from .views import (LandingPageView,
                    research_page_view)

app_name = 'mysite'

urlpatterns = [
    path('home/', LandingPageView.as_view(), name='test' ),
    path('research/', research_page_view, name='test' ),
]