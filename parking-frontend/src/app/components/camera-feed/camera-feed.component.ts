import { Component, Input, OnInit, OnDestroy, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ParkingService } from '../../services/parking.service';
import { ParkingStats } from '../../models/parking.model';
import { Subscription, interval, switchMap, startWith, catchError, of } from 'rxjs';

@Component({
  selector: 'app-camera-feed',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './camera-feed.component.html',
  styleUrls: ['./camera-feed.component.scss']
})
export class CameraFeedComponent implements OnInit, OnDestroy {
  @Input() parkingId: string = '';
  @Input() parkingName: string = '';

  videoUrl: string = '';
  stats: ParkingStats | null = null;
  isLoading: boolean = true;
  hasError: boolean = false;

  private statsSubscription?: Subscription;

  constructor(
    private parkingService: ParkingService,
    private cdr: ChangeDetectorRef
  ) {}

  ngOnInit(): void {
    // Générer l'URL du flux vidéo pour ce parking exmpel : http://localhost:5000/video/lac1
    this.videoUrl = this.parkingService.getVideoStreamUrl(this.parkingId);

    this.statsSubscription = interval(5000).pipe(
      startWith(0),
      switchMap(() => 
        this.parkingService.getLiveParkingStats(this.parkingId).pipe(
          catchError(err => {
            // 404 = pas encore prêt → retourner null silencieusement
            // Le polling continue automatiquement au prochain interval
            if (err.status === 404) {
              console.log(`⏳ ${this.parkingId} pas encore prêt, réessai dans 5s...`);
            } else {
              console.error(`❌ Erreur stats ${this.parkingId}:`, err);
            }
            return of(null);  // ← retourner null au lieu de planter
          })
        )
      )
    ).subscribe({
      next: (stats) => {
        if (stats === null) return;  // ← ignorer les nulls (404)
        
        // Ignorer si status = loading
        if ((stats as any).status === 'loading') {
          console.log(`⏳ ${this.parkingId} en cours de démarrage...`);
          return;
        }

        this.stats     = stats;
        this.isLoading = false;
        this.hasError  = false;
        this.cdr.detectChanges();
      }
    });
  }
  // Nettoyage de l'abonnement lors de la destruction du composant
  ngOnDestroy(): void {
    this.statsSubscription?.unsubscribe();
  }

  onImageLoad(): void {
    this.isLoading = false;
    this.hasError  = false;
  }

  onImageError(): void {
    this.isLoading = false;
    this.hasError  = true;
  }

  getOccupancyClass(): string {
    if (!this.stats) return '';
    const rate = this.stats.occupancy_rate;
    if (rate >= 90) return 'critical';
    if (rate >= 70) return 'warning';
    return 'normal';
  }
}