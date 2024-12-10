from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.views import LoginView as BaseLoginView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views import View
from django.shortcuts import render, redirect
from django.contrib import messages

from api.models import Usuario


class SignupView(BaseLoginView):
    redirect_authenticated_user = True

    def get(self, request):
        return render(request, 'signup.html')

    def post(self, request):
        username = request.POST.get('username')
        email = request.POST.get('email')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')

        if not all([email, password1, password2]):
            messages.error(request, 'All fields are required.')
            return render(request, 'signup.html')

        if password1 != password2:
            messages.error(request, 'Passwords do not match.')
            return render(request, 'signup.html')

        if Usuario.objects.filter(email=email).exists():
            messages.error(request, 'Email is already taken.')
            return render(request, 'signup.html')

        user = Usuario.objects.create_user(
            username=username, email=email, password=password1)
        login(request, user)
        return redirect('index-page')


class LoginView(BaseLoginView):
    redirect_authenticated_user = True

    def get(self, request):
        return render(request, 'login.html')

    def post(self, request):
        email = request.POST.get('email')
        password = request.POST.get('password')

        if not all([email, password]):
            messages.error(request, 'Email and password are required.')
            return render(request, 'login.html')

        user = authenticate(request, email=email, password=password)
        if user:
            login(request, user)
            return redirect('index-page')
        else:
            messages.error(request, 'Invalid Email or password.')
            return render(request, 'login.html')


class LogoutView(LoginRequiredMixin, View):
    redirect_authenticated_user = True

    def post(self, request):
        
        logout(request)
        return redirect('login')
