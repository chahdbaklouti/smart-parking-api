import { Component, OnInit, OnDestroy, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ParkingService } from '../../services/parking.service';
import { Camera, GlobalStats } from '../../models/parking.model';
import { CameraFeedComponent } from '../camera-feed/camera-feed.component';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, CameraFeedComponent],
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss']
})
export class DashboardComponent implements OnInit, OnDestroy {
  cameras: Camera[] = [];
  globalStats: GlobalStats | null = null;
  currentTime: Date = new Date();
  isLoading: boolean = true;
  error: string | null = null;

  private camerasSubscription?: Subscription;
  private statsSubscription?: Subscription;
  private timeInterval?: any;

  constructor(
    private parkingService: ParkingService,
    private cdr: ChangeDetectorRef
  ) {
    console.log('🏗️ Constructor DashboardComponent appelé');
  }

  ngOnInit(): void {
    console.log('🚀 ngOnInit DashboardComponent appelé');
    
    // Mettre à jour l'heure chaque seconde
    this.timeInterval = setInterval(() => {
      this.currentTime = new Date();
    }, 1000);

    // Charger la liste des caméras
    console.log('📞 Appel de loadCameras()...');
    this.loadCameras();
  }

  ngOnDestroy(): void {
    this.camerasSubscription?.unsubscribe();
    this.statsSubscription?.unsubscribe();
    if (this.timeInterval) {
      clearInterval(this.timeInterval);
    }
  }

  loadCameras(): void {
    this.isLoading = true;
    this.error = null;
    console.log('🔍 Tentative de chargement des caméras depuis:', this.parkingService['flaskUrl']);
    
    this.camerasSubscription = this.parkingService.getCameras().subscribe({
      next: (response) => {
        console.log('✅ Réponse reçue:', response);
        // Filtrer seulement les caméras avec annotations
        this.cameras = response.cameras.filter(c => c.hasAnnotation);
        console.log('📹 Caméras filtrées:', this.cameras);
        this.isLoading = false;
        console.log('🔄 isLoading après assignation:', this.isLoading);
        // Forcer la détection de changements
        this.cdr.detectChanges();
        console.log('✨ detectChanges() appelé');
      },
      error: (err) => {
        console.error('❌ Erreur chargement caméras:', err);
        this.error = 'Impossible de charger les caméras. Vérifiez que le serveur Flask est lancé.';
        this.isLoading = false;
        this.cdr.detectChanges();
      }
    });
  }

  // Formatage de l'heure
  formatTime(date: Date): string {
    return date.toLocaleTimeString('fr-FR', { 
      hour: '2-digit', 
      minute: '2-digit',
      second: '2-digit'
    });
  }

  // Formatage de la date
  formatDate(date: Date): string {
    return date.toLocaleDateString('fr-FR', { 
      weekday: 'long', 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric' 
    });
  }

  // Classe CSS selon le taux d'occupation
  getOccupancyClass(): string {
    if (!this.globalStats) return '';
    const rate = this.globalStats.global.occupancy_rate;
    if (rate >= 90) return 'critical';
    if (rate >= 70) return 'warning';
    return 'normal';
  }
}
