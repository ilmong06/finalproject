package com.example.wav2vecapp;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.location.Location;
import android.util.Log;
import android.widget.TextView;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.android.gms.location.*;

public class LocationHelper {

    private static final int LOCATION_PERMISSION_REQUEST_CODE = 1001;
    private final Activity activity;
    private final FusedLocationProviderClient fusedLocationClient;
    private final TextView textView;
    private final TextView textRegisterStep;

    public LocationHelper(Activity activity, TextView textView, TextView textRegisterStep) {
        this.activity = activity;
        this.textView = textView;
        this.textRegisterStep = textRegisterStep;
        this.fusedLocationClient = LocationServices.getFusedLocationProviderClient(activity);
    }

    public void requestLocationPermission() {
        if (ContextCompat.checkSelfPermission(activity, Manifest.permission.ACCESS_FINE_LOCATION)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(activity,
                    new String[]{Manifest.permission.ACCESS_FINE_LOCATION},
                    LOCATION_PERMISSION_REQUEST_CODE);
        } else {
            showCurrentLocation();
        }
    }

    public void onRequestPermissionsResult(int requestCode, int[] grantResults) {
        if (requestCode == LOCATION_PERMISSION_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                showCurrentLocation();
            } else {
                textView.setText("❌ 위치 권한이 거부되었습니다.");
            }
        }
    }

    private void showCurrentLocation() {
        Log.i("LocationHelper", "🔍 showCurrentLocation 호출됨");

        if (ActivityCompat.checkSelfPermission(activity, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED &&
                ActivityCompat.checkSelfPermission(activity, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            textView.setText("❗ 위치 권한이 없습니다.");
            return;
        }

        LocationRequest locationRequest = LocationRequest.create();
        locationRequest.setPriority(LocationRequest.PRIORITY_HIGH_ACCURACY);
        locationRequest.setInterval(3000);
        locationRequest.setFastestInterval(1000);
        locationRequest.setNumUpdates(1);

        LocationCallback locationCallback = new LocationCallback() {
            @Override
            public void onLocationResult(LocationResult locationResult) {
                if (locationResult == null || locationResult.getLastLocation() == null) {
                    textView.setText("❌ 위치를 가져올 수 없습니다. 다시 시도하세요.");
                    textRegisterStep.setText("📡 위치 요청 실패");
                    return;
                }

                Location location = locationResult.getLastLocation();
                double lat = location.getLatitude();
                double lon = location.getLongitude();
                String locationText = "📍 현재 위치\n위도: " + lat + "\n경도: " + lon;

                Log.i("LocationHelper", "✅ 위치 획득: " + locationText);
                textView.setText(locationText);
                textRegisterStep.setText(locationText);
            }
        };

        fusedLocationClient.requestLocationUpdates(locationRequest, locationCallback, activity.getMainLooper());
    }
}
