import { useState, useRef, useEffect } from 'react';
import {
  View, Text, TouchableOpacity, Image,
  StyleSheet, Alert, ActivityIndicator,
  SafeAreaView, StatusBar, Animated, Dimensions
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as MediaLibrary from 'expo-media-library';
import axios from 'axios';

const FLASK_URL = 'http://127.0.0.1:5050';
const { width } = Dimensions.get('window');

function SplashScreen({ onNext }) {
  const logoAnim = useRef(new Animated.Value(0)).current;
  const textAnim = useRef(new Animated.Value(0)).current;
  const btnAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.sequence([
      Animated.timing(logoAnim, { toValue: 1, duration: 800, useNativeDriver: true }),
      Animated.timing(textAnim, { toValue: 1, duration: 600, useNativeDriver: true }),
      Animated.timing(btnAnim, { toValue: 1, duration: 500, useNativeDriver: true }),
    ]).start();
  }, []);

  return (
    <SafeAreaView style={styles.splash}>
      <StatusBar barStyle="light-content" backgroundColor="#0d1f0f" />
      <Animated.Image
        source={require('./assets/sdd_logo.png')}
        style={[styles.logo, { opacity: logoAnim, transform: [{ scale: logoAnim }] }]}
        resizeMode="contain"
      />
      <Animated.View style={{ opacity: textAnim, alignItems: 'center' }}>
        <View style={styles.badge}>
          <Text style={styles.badgeText}>INDIAN ARMY</Text>
        </View>
        <Text style={styles.jaiHind}>
          <Text style={{ color: '#FF9933' }}>Jai </Text>
          <Text style={{ color: '#ffffff' }}>Hi</Text>
          <Text style={{ color: '#138808' }}>nd</Text>
        </Text>
        <Text style={styles.sddText}>Simulator Development Division</Text>
      </Animated.View>
      <Animated.View style={{ opacity: btnAnim, width: '100%' }}>
        <TouchableOpacity style={styles.mainBtn} onPress={onNext}>
          <Text style={styles.mainBtnText}>Next</Text>
        </TouchableOpacity>
      </Animated.View>
    </SafeAreaView>
  );
}

function InstructionsScreen({ onStart, onBack }) {
  const slideAnim = useRef(new Animated.Value(30)).current;
  const fadeAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(slideAnim, { toValue: 0, duration: 600, useNativeDriver: true }),
      Animated.timing(fadeAnim, { toValue: 1, duration: 600, useNativeDriver: true }),
    ]).start();
  }, []);

  return (
    <SafeAreaView style={styles.screen}>
      <StatusBar barStyle="light-content" backgroundColor="#0d1f0f" />
      <Animated.View style={[styles.centerContent, { opacity: fadeAnim, transform: [{ translateY: slideAnim }] }]}>
        <TouchableOpacity onPress={onBack} style={styles.backBtn}>
          <Text style={styles.backText}>← Back</Text>
        </TouchableOpacity>
        <View style={styles.badge}>
          <Text style={styles.badgeText}>INDIAN ARMY</Text>
        </View>
        <Text style={styles.title}>Try On the{'\n'}Army Uniform</Text>
        <Text style={styles.subtitle}>Stand back so your full body{'\n'}is visible in the camera</Text>
        <View style={styles.iconBox}>
          <Text style={styles.iconBoxIcon}>[ stand back ]</Text>
          <Text style={styles.iconBoxText}>full body must be in frame</Text>
        </View>
        <TouchableOpacity style={styles.mainBtn} onPress={onStart}>
          <Text style={styles.mainBtnText}>See Yourself in Army Fit</Text>
        </TouchableOpacity>
        <Text style={styles.privacy}>Your photo is deleted immediately after download</Text>
      </Animated.View>
    </SafeAreaView>
  );
}

function CameraScreen({ onCapture, onBack }) {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef(null);
  const [loading, setLoading] = useState(false);
  const [countdown, setCountdown] = useState(null);
  const fadeAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(fadeAnim, { toValue: 1, duration: 500, useNativeDriver: true }).start();
  }, []);

  const startCountdown = () => {
    setCountdown(3);
    let count = 3;
    const timer = setInterval(() => {
      count -= 1;
      if (count === 0) {
        clearInterval(timer);
        setCountdown(null);
        capture();
      } else {
        setCountdown(count);
      }
    }, 1000);
  };

  const capture = async () => {
    if (!cameraRef.current) return;
    setLoading(true);
    try {
      const photo = await cameraRef.current.takePictureAsync({ base64: true, quality: 0.8 });
      if (!photo || !photo.base64) throw new Error('Could not capture photo');
      const response = await axios.post(FLASK_URL + '/overlay', {
        image: photo.base64
      }, { timeout: 30000 });
      if (response.data.success) {
        onCapture(response.data.image);
      } else {
        Alert.alert('Error', response.data.error || 'Something went wrong');
      }
    } catch (err) {
      Alert.alert('Connection Error', 'Make sure Flask server is running on the laptop and both devices are on same WiFi.');
    } finally {
      setLoading(false);
    }
  };

  if (!permission || !permission.granted) {
    return (
      <SafeAreaView style={styles.screen}>
        <Text style={styles.title}>Camera Permission Needed</Text>
        <TouchableOpacity style={styles.mainBtn} onPress={requestPermission}>
          <Text style={styles.mainBtnText}>Grant Permission</Text>
        </TouchableOpacity>
      </SafeAreaView>
    );
  }

  return (
    <Animated.View style={{ flex: 1, opacity: fadeAnim }}>
      <CameraView ref={cameraRef} style={styles.camera} facing="back">
        <SafeAreaView style={styles.cameraUI}>
          <View style={styles.cameraTop}>
            <View style={styles.liveBadge}>
              <Text style={styles.liveText}>LIVE</Text>
            </View>
            <View style={styles.frontBadge}>
              <Text style={styles.frontText}>BACK</Text>
            </View>
          </View>
          {countdown !== null && (
            <View style={styles.countdownContainer}>
              <Text style={styles.countdownText}>{countdown}</Text>
            </View>
          )}
          <Text style={styles.cameraHint}>Make sure your full body is visible</Text>
          <View style={styles.cameraBottom}>
            {loading ? (
              <ActivityIndicator size="large" color="#ffffff" />
            ) : (
              <View style={{ alignItems: 'center' }}>
                <TouchableOpacity
                  style={styles.captureBtn}
                  onPress={startCountdown}
                  disabled={countdown !== null}
                >
                  <View style={styles.captureInner} />
                </TouchableOpacity>
                <Text style={styles.captureHint}>
                  {countdown !== null ? 'Taking photo in ' + countdown + '...' : 'Tap to capture'}
                </Text>
                <TouchableOpacity onPress={onBack} style={styles.backBtn}>
                  <Text style={styles.backText}>← Back</Text>
                </TouchableOpacity>
              </View>
            )}
          </View>
        </SafeAreaView>
      </CameraView>
    </Animated.View>
  );
}

function ResultScreen({ image, onRetry }) {
  const [mediaPermission, requestMediaPermission] = MediaLibrary.usePermissions();
  const scaleAnim = useRef(new Animated.Value(0.9)).current;
  const fadeAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(scaleAnim, { toValue: 1, duration: 500, useNativeDriver: true }),
      Animated.timing(fadeAnim, { toValue: 1, duration: 500, useNativeDriver: true }),
    ]).start();
  }, []);

  const saveImage = async () => {
    if (!mediaPermission || !mediaPermission.granted) {
      await requestMediaPermission();
    }
    try {
      await MediaLibrary.saveToLibraryAsync('data:image/jpeg;base64,' + image);
      Alert.alert('Saved!', 'Image saved to your gallery.');
    } catch (e) {
      Alert.alert('Error', 'Could not save image.');
    }
  };

  return (
    <SafeAreaView style={styles.screen}>
      <StatusBar barStyle="light-content" backgroundColor="#0d1f0f" />
      <Animated.View style={[styles.centerContent, { opacity: fadeAnim, transform: [{ scale: scaleAnim }] }]}>
        <Text style={styles.resultTitle}>Your Army Look</Text>
        <Image
          source={{ uri: 'data:image/jpeg;base64,' + image }}
          style={styles.resultImage}
          resizeMode="contain"
        />
        <TouchableOpacity style={styles.mainBtn} onPress={saveImage}>
          <Text style={styles.mainBtnText}>Save to Gallery</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.outlineBtn} onPress={onRetry}>
          <Text style={styles.outlineBtnText}>Try Again</Text>
        </TouchableOpacity>
        <Text style={styles.privacy}>Photo auto-deletes from server after demo expiry</Text>
      </Animated.View>
    </SafeAreaView>
  );
}

export default function App() {
  const [screen, setScreen] = useState('splash');
  const [resultImage, setResultImage] = useState(null);

  if (screen === 'splash') {
    return <SplashScreen onNext={() => setScreen('instructions')} />;
  }
  if (screen === 'instructions') {
    return <InstructionsScreen onStart={() => setScreen('camera')} onBack={() => setScreen('splash')} />;
  }
  if (screen === 'camera') {
    return (
      <CameraScreen
        onCapture={(img) => { setResultImage(img); setScreen('result'); }}
        onBack={() => setScreen('instructions')}
      />
    );
  }
  if (screen === 'result') {
    return (
      <ResultScreen
        image={resultImage}
        onRetry={() => setScreen('camera')}
      />
    );
  }
  return null;
}

const styles = StyleSheet.create({
  splash: { flex: 1, backgroundColor: '#0d1f0f', alignItems: 'center', justifyContent: 'space-around', padding: 32 },
  screen: { flex: 1, backgroundColor: '#0d1f0f', alignItems: 'center', justifyContent: 'center', padding: 24 },
  centerContent: { width: '100%', alignItems: 'center' },
  logo: { width: 180, height: 180 },
  badge: { backgroundColor: '#1a3a1a', borderRadius: 6, paddingHorizontal: 14, paddingVertical: 5, marginBottom: 14 },
  badgeText: { color: '#4a8c3a', fontSize: 11, fontWeight: '600', letterSpacing: 2 },
  jaiHind: { fontSize: 36, fontWeight: '700', marginBottom: 8 },
  sddText: { color: '#2a4a2a', fontSize: 12, marginBottom: 24 },
  title: { color: '#ffffff', fontSize: 28, fontWeight: '700', textAlign: 'center', marginBottom: 12, lineHeight: 36 },
  subtitle: { color: '#4a6a4a', fontSize: 14, textAlign: 'center', marginBottom: 24, lineHeight: 22 },
  iconBox: { backgroundColor: '#1a2a1a', borderRadius: 10, padding: 16, width: '100%', alignItems: 'center', marginBottom: 24, borderWidth: 0.5, borderColor: '#2a4a2a' },
  iconBoxIcon: { color: '#4a8c3a', fontSize: 14, marginBottom: 4 },
  iconBoxText: { color: '#3a6a3a', fontSize: 11 },
  mainBtn: { backgroundColor: '#3a6a2a', borderRadius: 14, paddingVertical: 16, width: '100%', alignItems: 'center', marginBottom: 12 },
  mainBtnText: { color: '#ffffff', fontSize: 16, fontWeight: '700' },
  outlineBtn: { borderRadius: 14, paddingVertical: 14, width: '100%', alignItems: 'center', marginBottom: 12, borderWidth: 0.5, borderColor: '#2a4a2a' },
  outlineBtnText: { color: '#4a6a4a', fontSize: 15 },
  privacy: { color: '#2a4a2a', fontSize: 11, textAlign: 'center', marginTop: 4 },
  camera: { flex: 1 },
  cameraUI: { flex: 1, justifyContent: 'space-between', padding: 16 },
  cameraTop: { flexDirection: 'row', justifyContent: 'space-between' },
  liveBadge: { backgroundColor: 'rgba(58,106,42,0.85)', borderRadius: 5, paddingHorizontal: 8, paddingVertical: 3 },
  liveText: { color: '#e8f0e8', fontSize: 10, fontWeight: '600' },
  frontBadge: { backgroundColor: 'rgba(0,0,0,0.6)', borderRadius: 5, paddingHorizontal: 8, paddingVertical: 3 },
  frontText: { color: '#4a8c3a', fontSize: 10 },
  cameraHint: { backgroundColor: 'rgba(0,0,0,0.5)', color: '#ffffff', textAlign: 'center', padding: 8, borderRadius: 8, fontSize: 12 },
  cameraBottom: { alignItems: 'center', paddingBottom: 16 },
  captureBtn: { width: 72, height: 72, borderRadius: 36, borderWidth: 2.5, borderColor: '#ffffff', alignItems: 'center', justifyContent: 'center', marginBottom: 8 },
  captureInner: { width: 56, height: 56, borderRadius: 28, backgroundColor: '#ffffff' },
  captureHint: { color: '#aaaaaa', fontSize: 12, marginBottom: 12 },
  backBtn: { paddingVertical: 8, paddingHorizontal: 24 },
  backText: { color: '#4a6a4a', fontSize: 14 },
  countdownContainer: { position: 'absolute', top: '35%', left: 0, right: 0, alignItems: 'center' },
  countdownText: { color: '#ffffff', fontSize: 90, fontWeight: '700' },
  resultTitle: { color: '#ffffff', fontSize: 22, fontWeight: '700', marginBottom: 16 },
  resultImage: { width: width - 48, height: 400, borderRadius: 14, marginBottom: 20 },
});